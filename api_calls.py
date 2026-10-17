"""LLM calls, interactive and batch.

Routing is by model name: ``gpt*`` to OpenAI and ``gemini*`` to Vertex through
``vertex.py``. Any other name raises.
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
import openai

load_dotenv()


class _LazyClient:
    """Builds the SDK client on first use, so importing this module needs no
    API keys."""

    def __init__(self, factory):
        self._factory = factory
        self._client = None

    def __getattr__(self, name):
        if self._client is None:
            self._client = self._factory()
        return getattr(self._client, name)


# OPENAI_API_KEY. The SDK retries connection errors
# and 408/409/429/5xx responses this many times, including batch polling.
client_openai = _LazyClient(
    lambda: openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=5,
    )
)
# Explicit timeout for the long Responses API tool loop.
TOOL_LOOP_TIMEOUT = 60 * 60

#: Gemini 2.5 output ceiling; thinking tokens share it.
GEMINI_MAX_OUTPUT_TOKENS = 65_536


class TruncatedOutput(RuntimeError):
    """The model stopped because it hit max_completion_tokens."""


def _messages(content, system_prompt=""):
    """Chat-completions messages. The system turn is omitted when empty."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _normalize_openai_response_format(response_format):
    """Convert common schema shorthands into OpenAI's response_format shape."""
    if "type" in response_format and response_format["type"] in (
        "json_object",
        "json_schema",
    ):
        return response_format

    json_schema_payload = {
        "name": "response",
        "strict": True,
        "schema": response_format,
    }

    if (
        "schema" in response_format
        and isinstance(response_format["schema"], dict)
        and "name" in response_format
    ):
        json_schema_payload["name"] = response_format["name"]
        json_schema_payload["schema"] = response_format["schema"]
        if "strict" in response_format:
            json_schema_payload["strict"] = response_format["strict"]

    return {"type": "json_schema", "json_schema": json_schema_payload}


# ── Interactive calls ─────────────────────────────────────────────────────────


def query_openai(
    content,
    system_prompt="",
    model="",
    response_format=None,
    temperature=1,
    max_completion_tokens=128_000,
):
    """One OpenAI chat completion; the parsed JSON when ``response_format`` is
    set, else the text.

    Raises RuntimeError on an empty reply and ``TruncatedOutput`` when
    ``finish_reason`` is ``"length"``.
    """
    kwargs = dict(
        model=model,
        messages=_messages(content, system_prompt),
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )
    if response_format is not None:
        kwargs["response_format"] = _normalize_openai_response_format(
            response_format
        )
    response = client_openai.chat.completions.create(**kwargs)
    choice = response.choices[0]
    text = choice.message.content
    if not text:
        raise RuntimeError(
            f"OpenAI returned empty content (finish_reason={choice.finish_reason!r})."
        )
    if choice.finish_reason == "length":
        raise TruncatedOutput(
            f"""OpenAI truncated output (finish_reason='length', max_completion_tokens={max_completion_tokens})."""
        )
    return json.loads(text) if response_format is not None else text


def query_vertex(
    content,
    system_prompt="",
    model="",
    response_format=None,
    max_completion_tokens=GEMINI_MAX_OUTPUT_TOKENS,
):
    """One synchronous Gemini call on Vertex, thinking on (dynamic budget)."""
    import vertex

    schema = (
        vertex.unwrap_response_format(response_format)
        if response_format
        else None
    )
    by_key = vertex.run_requests(
        [("0", content)],
        model=model,
        thinking_budget=-1,
        mode="sync",
        system_instruction=system_prompt or None,
        response_schema=schema,
        max_output_tokens=min(
            int(max_completion_tokens), GEMINI_MAX_OUTPUT_TOKENS
        ),
    )
    text = by_key["0"]
    if not text:
        raise RuntimeError(f"empty Vertex result from {model}")
    return json.loads(text) if response_format else text


def query_llm(
    content,
    system_prompt="",
    model="",
    response_format=None,
    temperature=1,
    max_completion_tokens=128_000,
    cache_prefix=None,
):
    """Route one LLM call by model name (``gpt*`` or ``gemini*``).
    """
    if cache_prefix:
        content = f"{cache_prefix}\n\n{content}"
    if "gemini" in model:
        return query_vertex(
            content,
            system_prompt,
            model=model,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
        )
    if "gpt" in model:
        return query_openai(
            content,
            system_prompt,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
    raise ValueError(f"unsupported model {model!r}")


# ── Tool loop (OpenAI Responses API) ──────────────────────────────────────────

# OpenAI hosted web search.
OPENAI_WEB_SEARCH_TOOL = {"type": "web_search"}


def _nullable_schema(spec):
    spec = dict(spec)
    if "type" in spec:
        t = spec["type"]
        if t == "null" or (isinstance(t, list) and "null" in t):
            return spec
        spec["type"] = (
            list(t) + ["null"] if isinstance(t, list) else [t, "null"]
        )
        if "enum" in spec:
            spec["enum"] = list(spec["enum"]) + [None]
        return spec
    return {"anyOf": [spec, {"type": "null"}]}


def _openai_strict_parameters(parameters):
    """Every property must be required for OpenAI strict function tools."""
    parameters = dict(parameters)
    properties = (
        dict(parameters["properties"]) if "properties" in parameters else {}
    )
    originally_required = (
        set(parameters["required"]) if "required" in parameters else set()
    )
    strict_properties = {}
    for key, spec in properties.items():
        if key in originally_required:
            strict_properties[key] = spec
        else:
            strict_properties[key] = _nullable_schema(spec)
    parameters["properties"] = strict_properties
    parameters["required"] = list(properties)
    parameters["additionalProperties"] = False
    return parameters


def _openai_tool_defs(tools, web_search=False):
    """Responses API tool list, with OpenAI's hosted web_search first if asked."""
    defs = []
    if web_search:
        defs.append(dict(OPENAI_WEB_SEARCH_TOOL))
    for spec in tools:
        defs.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec["description"],
                "parameters": _openai_strict_parameters(spec["parameters"]),
                "strict": True,
            }
        )
    return defs


def _missing_required(args, schema):
    required = schema["required"] if "required" in schema else []
    return [key for key in required if key not in args]


def _terminal_schema(tools, terminal_tool):
    for spec in tools:
        if spec["name"] == terminal_tool:
            return spec["parameters"]
    raise ValueError(f"no tool named {terminal_tool!r}")


def _incomplete_terminal_result(name, missing):
    print(f"[tools] {name} rejected: missing {', '.join(missing)}", flush=True)
    return json.dumps(
        {
            "error": f"{name} missing required fields",
            "missing": missing,
        },
        ensure_ascii=False,
    )


def _log_tool_call(name, args):
    payload = json.dumps(args, ensure_ascii=False)
    if len(payload) > 200:
        payload = payload[:199] + "…"
    print(f"[tools] {name} {payload}", flush=True)


def _item_field(item, name, default=None):
    if isinstance(item, dict):
        return item[name] if name in item else default
    return getattr(item, name, default)


def _web_search_log_args(item):
    action = _item_field(item, "action")
    if action is None:
        return {}
    atype = _item_field(action, "type")
    queries = _item_field(action, "queries")
    query = _item_field(action, "query")
    url = _item_field(action, "url")
    args = {}
    if atype:
        args["action"] = atype
    if queries:
        args["queries"] = list(queries)
    elif query:
        args["query"] = query
    if url:
        args["url"] = url
    return args


def _log_openai_output(output):
    parts = []
    for item in output:
        itype = _item_field(item, "type", "?")
        if itype == "function_call":
            parts.append(f"tool_use:{_item_field(item, 'name')}")
        elif itype == "web_search_call":
            parts.append("web_search")
        else:
            parts.append(str(itype))
    print(
        f"[tools] blocks: {', '.join(parts) if parts else '(none)'}",
        flush=True,
    )


def query_llm_with_tools(
    content,
    system_prompt="",
    model="",
    tools=None,
    execute=None,
    terminal_tool="submit_revision",
    web_search=False,
    temperature=1,
    max_completion_tokens=128_000,
):
    """Run a client-tool loop and return the arguments of ``terminal_tool``.
    """
    tool_defs = _openai_tool_defs(tools, web_search=web_search)
    terminal_schema = _terminal_schema(tools, terminal_tool)
    previous_response_id = None
    pending_input = content
    while True:
        kwargs = dict(
            model=model,
            input=pending_input,
            tools=tool_defs,
            temperature=temperature,
            max_output_tokens=max_completion_tokens,
        )
        if system_prompt:
            kwargs["instructions"] = system_prompt
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        response = client_openai.responses.create(
            timeout=TOOL_LOOP_TIMEOUT, **kwargs
        )
        previous_response_id = response.id
        output = list(response.output)
        _log_openai_output(output)
        function_outputs = []
        for item in output:
            itype = _item_field(item, "type")
            if itype == "web_search_call":
                _log_tool_call("web_search", _web_search_log_args(item))
                continue
            if itype != "function_call":
                continue
            name = _item_field(item, "name")
            raw = _item_field(item, "arguments")
            args = json.loads(raw) if raw else {}
            _log_tool_call(name, args)
            if name == terminal_tool:
                missing = _missing_required(args, terminal_schema)
                if not missing:
                    return args
                result = _incomplete_terminal_result(name, missing)
            else:
                result = execute(name, args)
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False)
            function_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": _item_field(item, "call_id"),
                    "output": result,
                }
            )
        if not function_outputs:
            raise RuntimeError(
                f"""OpenAI ended the tool loop without calling {terminal_tool!r} (status={response.status!r})."""
            )
        pending_input = function_outputs


def create_embedding(content, model="text-embedding-3-large"):
    """Embed ``content`` with OpenAI and return the vector."""
    response = client_openai.embeddings.create(input=content, model=model)
    return response.data[0].embedding


# ── Batch ─────────────────────────────────────────────────────────────────────


def _make_custom_id(index: int, prefix: str = "req") -> str:
    """Stable, unique custom_id for a batch request."""
    return f"{prefix}-{index:06d}"


def _parse_index(custom_id: str) -> int:
    """Reverse of _make_custom_id: extract the integer index."""
    return int(custom_id.split("-")[-1])


def _maybe_parse_json_content(content: Any) -> Any:
    """The parsed JSON object or array in ``content``, else ``content``
    unchanged (for example output cut off mid-object)."""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return content
    return content


def _format_error_payload(payload: Any) -> str:
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def _parse_openai_batch_files(
    output_text: str,
    error_text: str,
    response_format: dict[str, Any] | None = None,
) -> tuple[dict[int, Any], list[str], set[int]]:
    """Parse OpenAI batch output and error JSONL.

    Returns ``(results_by_index, errors, error_indices)``.
    """
    results: dict[int, Any] = {}
    errors: list[str] = []
    error_indices: set[int] = set()

    for line in output_text.strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        idx = _parse_index(custom_id)
        if record["error"]:
            errors.append(
                f"{custom_id}: {_format_error_payload(record['error'])}"
            )
            error_indices.add(idx)
            continue
        message = record["response"]["body"]["choices"][0]["message"]["content"]
        results[idx] = (
            _maybe_parse_json_content(message)
            if response_format is not None
            else message
        )

    for line in error_text.strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = record["custom_id"]
        error_indices.add(_parse_index(custom_id))
        errors.append(
            f"""{custom_id}: {_format_error_payload(record['error'] if record['error'] else record)}"""
        )

    return results, errors, error_indices


def submit_openai_batch(
    items: list[dict[str, Any]],
    model: str,
    response_format: dict | None = None,
    temperature: float = 1.0,
    max_completion_tokens: int = 128_000,
    custom_id_prefix: str = "req",
) -> str:
    """Create an OpenAI Batch. Does not poll."""
    lines: list[str] = []
    for i, item in enumerate(items):
        content = item["content"]
        if "cache_prefix" in item and item["cache_prefix"]:
            content = f"{item['cache_prefix']}\n\n{content}"
        body: dict[str, Any] = dict(
            model=model,
            messages=_messages(
                content,
                item["system_prompt"] if "system_prompt" in item else "",
            ),
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )
        if response_format is not None:
            body["response_format"] = _normalize_openai_response_format(
                response_format
            )
        lines.append(
            json.dumps(
                {
                    "custom_id": _make_custom_id(i, custom_id_prefix),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        )

    file_obj = client_openai.files.create(
        file=(
            "batch_input.jsonl",
            io.BytesIO("\n".join(lines).encode()),
            "application/json",
        ),
        purpose="batch",
    )
    print(f"[OpenAI batch] uploaded input file → {file_obj.id}")
    batch = client_openai.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"[OpenAI batch] submitted {len(lines)} requests → {batch.id}")
    return batch.id


def batch_query_openai(
    items: list[dict[str, Any]],
    model: str,
    response_format: dict | None = None,
    temperature: float = 1.0,
    max_completion_tokens: int = 128_000,
    poll_interval: int = 60,
    custom_id_prefix: str = "req",
) -> list[tuple[str, Any]]:
    """Submit *items* as one OpenAI Batch (chat/completions) and block.

    Returns ``(custom_id, response)`` in input order. A failed or missing
    request raises.
    """
    batch_id = submit_openai_batch(
        items,
        model=model,
        response_format=response_format,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        custom_id_prefix=custom_id_prefix,
    )

    while True:
        batch = client_openai.batches.retrieve(batch_id)
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            break
        counts = batch.request_counts
        print(
            f"""[OpenAI batch] {batch_id} {batch.status} — total={counts.total} completed={counts.completed} failed={counts.failed}"""
        )
        time.sleep(poll_interval)

    if batch.status != "completed":
        raise RuntimeError(f"[OpenAI batch] ended with status '{batch.status}'")

    output_text = (
        client_openai.files.content(batch.output_file_id).text
        if batch.output_file_id
        else ""
    )
    error_text = (
        client_openai.files.content(batch.error_file_id).text
        if batch.error_file_id
        else ""
    )

    results, errors, error_indices = _parse_openai_batch_files(
        output_text=output_text,
        error_text=error_text,
        response_format=response_format,
    )
    missing_indices = [
        i
        for i in range(len(items))
        if i not in results and i not in error_indices
    ]
    problems = errors + [
        f"{_make_custom_id(i, custom_id_prefix)}: missing output record"
        for i in missing_indices
    ]
    if problems:
        raise RuntimeError(
            f"[OpenAI batch] {len(problems)} request(s) failed or missing:\n"
            + "\n".join(problems)
        )
    return [
        (_make_custom_id(i, custom_id_prefix), results[i])
        for i in range(len(items))
    ]


def batch_query_vertex(
    items: list[dict[str, Any]],
    model: str,
    response_format: dict | None = None,
    custom_id_prefix: str = "req",
) -> list[tuple[str, Any]]:
    """Run *items* as Vertex batch prediction and block.

    ``vertex.run_requests`` takes one system instruction per job, so items are
    grouped by ``system_prompt`` and each group is one run, stored under a GCS
    prefix named by a hash of the model, schema, system prompt and requests.

    Returns ``(custom_id, response)`` in input order. With a ``response_format``
    the response is the parsed JSON object, as on the OpenAI path. A request that
    returned nothing raises.
    """
    import hashlib
    import vertex

    schema = (
        vertex.unwrap_response_format(response_format)
        if response_format
        else None
    )

    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        system_prompt = item["system_prompt"] if "system_prompt" in item else ""
        groups.setdefault(system_prompt, []).append(index)

    out: dict[int, tuple[str, Any]] = {}
    for system_prompt, indices in groups.items():
        requests = []
        digest = hashlib.sha256()
        digest.update(
            json.dumps([model, schema, system_prompt], sort_keys=True).encode()
        )
        for i in indices:
            item = items[i]
            text = item["content"]
            if "cache_prefix" in item and item["cache_prefix"]:
                text = f"{item['cache_prefix']}\n\n{text}"
            requests.append((str(i), text))
            digest.update(text.encode())
        print(f"[vertex] {model}: {len(requests)} request(s)")
        by_key = vertex.run_requests(
            requests,
            model=model,
            thinking_budget=-1,
            mode="batch",
            system_instruction=system_prompt or None,
            response_schema=schema,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            gcs_prefix=f"batch_query_llm/{digest.hexdigest()[:16]}",
        )
        for i in indices:
            text = by_key[str(i)]
            if not text:
                raise RuntimeError(f"[vertex] request {i} returned nothing")
            out[i] = (
                _make_custom_id(i, custom_id_prefix),
                _maybe_parse_json_content(text) if schema else text,
            )
    return [out[i] for i in range(len(items))]


def batch_query_llm(
    items: list[dict[str, Any]],
    model: str = "",
    response_format: dict | None = None,
    temperature: float = 1.0,
    max_completion_tokens: int = 128_000,
    poll_interval: int = 60,
    custom_id_prefix: str = "req",
) -> list[tuple[str, Any]]:
    """Batch equivalent of ``query_llm``.
    """
    if "gemini" in model:
        return batch_query_vertex(
            items,
            model=model,
            response_format=response_format,
            custom_id_prefix=custom_id_prefix,
        )
    if "gpt" in model:
        return batch_query_openai(
            items,
            model=model,
            response_format=response_format,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            poll_interval=poll_interval,
            custom_id_prefix=custom_id_prefix,
        )
    raise ValueError(f"unsupported model {model!r}")
