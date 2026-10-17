"""Gemini calls on Vertex AI, as a Batch Prediction job (GCS input and output,
asynchronous) or synchronously (generate_content). Used for the Gemini 2.5 Pro
annotator and LLM-as-judge calls.

Auth is Vertex-only (no API keys). It uses Application Default Credentials and,
when GOOGLE_IMPERSONATE_SERVICE_ACCOUNT is set, impersonates that service
account. Environment:
  GOOGLE_CLOUD_PROJECT               (required)
  GOOGLE_CLOUD_LOCATION              (default us-central1)
  GOOGLE_IMPERSONATE_SERVICE_ACCOUNT (optional)
  GCS_BATCH_BUCKET                   (batch only) a bucket the account can
                                     read and write, e.g. gs://<bucket>

``run_requests`` takes ``(key, prompt)`` pairs and returns ``{key: text}``;
text is None for a request that returned nothing. A batch is one job. Each
request carries its key as a Gemini request label, so outputs map back to keys
by label, not by order.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from google import genai
from google.genai import types

POLL_INTERVAL = 30
# Response cap. It is only a ceiling; billing is by tokens generated.
MAX_OUTPUT_TOKENS = 32_768
# Give the batch job up to ~26h before giving up.
BATCH_TIMEOUT = 26 * 3600
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_SUCCEEDED = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
_TERMINAL_BAD = {
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PAUSED",
}


def unwrap_response_format(response_format):
    """The bare JSON Schema from an OpenAI-style envelope.

    The envelope is ``{"name", "schema", "strict"}``. Vertex's
    ``response_schema`` is the schema itself; handed the envelope it describes
    an object with properties ``name``/``schema``/``strict`` and every call
    comes back unusable.
    """
    body = (
        response_format["schema"]
        if "schema" in response_format
        and isinstance(response_format["schema"], dict)
        else response_format
    )
    return {k: v for k, v in body.items() if k not in ("strict", "name")}


def _credentials():
    """ADC, impersonating GOOGLE_IMPERSONATE_SERVICE_ACCOUNT when set."""
    import google.auth
    from google.auth import impersonated_credentials

    source, _ = google.auth.default(scopes=SCOPES)
    target = os.getenv("GOOGLE_IMPERSONATE_SERVICE_ACCOUNT")
    if not target:
        return source
    return impersonated_credentials.Credentials(
        source_credentials=source, target_principal=target, target_scopes=SCOPES
    )


def make_client():
    """A google-genai Client pointed at Vertex AI."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise EnvironmentError(
            "GOOGLE_CLOUD_PROJECT not set (Vertex requires a project)."
        )
    return genai.Client(
        vertexai=True,
        project=project,
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        credentials=_credentials(),
    )


def _gemini_schema(schema):
    """Drop ``additionalProperties``. Gemini's response schema rejects that
    keyword."""
    if isinstance(schema, dict):
        return {
            key: _gemini_schema(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_gemini_schema(item) for item in schema]
    return schema


def _content_config(
    thinking_budget, max_output_tokens, system_instruction, response_schema
):
    kwargs = dict(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        max_output_tokens=max_output_tokens,
    )
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
        kwargs["response_mime_type"] = "application/json"
        kwargs["response_schema"] = _gemini_schema(response_schema)
    return types.GenerateContentConfig(**kwargs)


def _run_sync(requests, client, model, cfg):
    results = {}
    for key, prompt in requests:
        text = client.models.generate_content(
            model=model, contents=prompt, config=cfg
        ).text
        results[key] = text if text and text.strip() else None
    return results


# ── batch ─────────────────────────────────────────────────────────────────────


def _label(key):
    # Vertex coerces numeric-looking label values to int then errors; wrap to
    # force a string.
    return f"/{key}/"


def _unlabel(v):
    return str(v).strip("/")


def _build_input_line(
    key,
    prompt,
    thinking_budget,
    system_instruction=None,
    response_schema=None,
    max_output_tokens=MAX_OUTPUT_TOKENS,
):
    """One Vertex batch input line.

    A GenerateContent request carrying the key as a label so the output maps
    back to it. camelCase to match the REST schema Vertex reads from GCS.
    """
    generation = {
        "thinkingConfig": {"thinkingBudget": thinking_budget},
        "maxOutputTokens": max_output_tokens,
    }
    if response_schema is not None:
        generation["responseMimeType"] = "application/json"
        generation["responseSchema"] = _gemini_schema(response_schema)
    request = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation,
        "labels": {"key": _label(key)},
    }
    if system_instruction:
        request["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return json.dumps({"request": request})


def _status_text(status):
    if isinstance(status, dict):
        if "message" in status and status["message"]:
            return str(status["message"])
        return json.dumps(status)
    return str(status).strip()


def _parse_output_line(obj):
    """One Vertex batch output line -> ``(key, text, error)``.

    Output lines echo the request (including labels) and add a ``response``
    and/or an error status. ``text`` is None when there is no usable text.
    """
    key = _unlabel(obj["request"]["labels"]["key"])
    if "status" in obj and str(obj["status"]).strip():
        return key, None, _status_text(obj["status"])
    response = obj["response"] if "response" in obj else {}
    candidates = response["candidates"] if "candidates" in response else []
    text = None
    if candidates:
        content = candidates[0]["content"] if "content" in candidates[0] else {}
        parts = content["parts"] if "parts" in content else []
        text = "".join(p["text"] for p in parts if "text" in p) or None
    return key, text, None if text else "empty batch result"


def _iter_jsonl(text):
    """Yield each JSON object in a Vertex predictions file.

    NOT line-based: Vertex pretty-prints larger records across multiple
    physical lines, so ``splitlines()`` plus ``json.loads`` would shatter them.
    ``raw_decode`` consumes one complete object at a time regardless of
    embedded newlines or whitespace between objects.
    """
    decoder = json.JSONDecoder()
    i, n = 0, len(text)
    while True:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            return
        obj, i = decoder.raw_decode(text, i)
        yield obj


def _is_final_batch_output(name):
    """A finished predictions JSONL, not a partial shard written while the job
    runs.

    Vertex writes those shards under an ``incremental_predictions`` directory.
    Match that directory as its own path component. A prefix that only contains
    the word ``incremental`` is still the final file.
    """
    if not name.endswith(".jsonl"):
        return False
    return "incremental_predictions" not in name.split("/")


def _run_batch(
    requests,
    client,
    model,
    thinking_budget,
    system_instruction,
    response_schema,
    max_output_tokens,
    gcs_prefix,
):
    from google.cloud import storage

    bucket_env = os.getenv("GCS_BATCH_BUCKET")
    if not bucket_env:
        raise EnvironmentError(
            "GCS_BATCH_BUCKET not set (required for Vertex batch mode)."
        )
    bucket_name = bucket_env.replace("gs://", "").split("/")[0]
    gcs = storage.Client(
        project=os.getenv("GOOGLE_CLOUD_PROJECT"), credentials=_credentials()
    )
    bucket = gcs.bucket(bucket_name)

    job_uuid = uuid.uuid4().hex
    prefix = gcs_prefix.strip("/")
    in_path = f"{prefix}/{job_uuid}/input.jsonl"
    out_prefix = f"{prefix}/{job_uuid}/output/"
    body = "\n".join(
        _build_input_line(
            key,
            prompt,
            thinking_budget,
            system_instruction=system_instruction,
            response_schema=response_schema,
            max_output_tokens=max_output_tokens,
        )
        for key, prompt in requests
    )
    bucket.blob(in_path).upload_from_string(
        body + "\n", content_type="application/jsonl"
    )
    job = client.batches.create(
        model=model,
        src=f"gs://{bucket_name}/{in_path}",
        config=types.CreateBatchJobConfig(
            dest=f"gs://{bucket_name}/{out_prefix}"
        ),
    )
    print(f"  vertex batch: {len(requests):,} requests -> {job.name}")

    start = time.time()
    while True:
        job = client.batches.get(name=job.name)
        state = job.state.name
        if state in _SUCCEEDED:
            break
        if state in _TERMINAL_BAD:
            raise RuntimeError(
                f"Vertex batch job {job.name} ended {state}: {job.error}"
            )
        if time.time() - start > BATCH_TIMEOUT:
            raise TimeoutError(f"Vertex batch timed out after {BATCH_TIMEOUT}s")
        time.sleep(POLL_INTERVAL)

    out_uri = job.dest.gcs_uri
    out_bucket = out_uri.split("/")[2]
    out_path = "/".join(out_uri.split("/")[3:])
    results = {key: None for key, _ in requests}
    for blob in gcs.bucket(out_bucket).list_blobs(prefix=out_path):
        if not _is_final_batch_output(blob.name):
            continue
        for obj in _iter_jsonl(blob.download_as_text()):
            key, text, error = _parse_output_line(obj)
            if key not in results:
                continue
            results[key] = text
            if error:
                print(f"  vertex batch: request {key}: {error}")
    return results


def run_requests(
    requests,
    *,
    model,
    thinking_budget,
    mode="batch",
    system_instruction=None,
    response_schema=None,
    max_output_tokens=MAX_OUTPUT_TOKENS,
    gcs_prefix="pidpipe",
):
    """Run ``(key, prompt)`` requests via Vertex Gemini; ``{key: text}``.

    ``mode="batch"`` is one Batch Prediction job; ``mode="sync"`` is one
    ``generate_content`` call per request. ``gcs_prefix`` is the object prefix
    inside ``GCS_BATCH_BUCKET``.
    """
    requests = list(requests)
    if not requests:
        return {}
    client = make_client()
    if mode == "sync":
        cfg = _content_config(
            thinking_budget,
            max_output_tokens,
            system_instruction,
            response_schema,
        )
        return _run_sync(requests, client, model, cfg)
    if mode == "batch":
        return _run_batch(
            requests,
            client,
            model,
            thinking_budget,
            system_instruction,
            response_schema,
            max_output_tokens,
            gcs_prefix,
        )
    raise ValueError(f"unknown mode {mode!r} (expected 'sync' or 'batch')")
