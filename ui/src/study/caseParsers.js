export function primaryCitation(raw) {
  if (!raw) return null;
  const text = String(raw).trim();
  let cites = text.startsWith("[") ? JSON.parse(text) : text.split(";");
  cites = cites.map((cite) => String(cite ?? "").trim()).filter(Boolean);
  return cites.find((cite) => !/\b(WL|LEXIS)\b/.test(cite)) ?? cites[0] ?? null;
}

export function parseParagraphsExport(rawJson) { return parseParagraphs(rawJson); }
export function parseSegmentationExport(rawJson) { return parseSegmentationPayload(rawJson); }
export function parseRelevanceExport(rawJson) { return parseRelevancePayload(rawJson); }
export function parseMajorityBoundaryExport(rawJson) { return parseMajorityBoundary(rawJson); }

function parseParagraphs(rawJson) {
  if (!rawJson) {
    return [];
  }
  try {
    const parsed = JSON.parse(rawJson);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.map((entry) => String(entry ?? "").trim()).filter((entry) => entry.length > 0);
  } catch {
    return [];
  }
}

function parseRelevancePayload(rawJson) {
  if (!rawJson) {
    return null;
  }
  try {
    const parsed = JSON.parse(rawJson);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function parseSegmentationPayload(rawJson) {
  if (!rawJson) {
    return [];
  }
  try {
    const parsed = JSON.parse(rawJson);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed;
  } catch {
    return [];
  }
}

/**
 * Normalize `majority_boundary_json`, written by isolate_majority_paragraphs in
 * llm_1_tree_extraction/segmenter.py.
 *
 * `endIndex` is exclusive: paragraphs before it are majority text. The paragraph at
 * `endIndex` is the one carrying the concurrence/dissent byline; when `splitBoundary`
 * is set, its first `endCharOffset` characters are still majority text and were sent
 * to the models, and the remainder was cut.
 */
function parseMajorityBoundary(rawJson) {
  if (!rawJson) {
    return null;
  }
  try {
    const parsed = typeof rawJson === "string" ? JSON.parse(rawJson) : rawJson;
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    const endIndex = Number(parsed.end_index);
    const endCharOffset = Number(parsed.end_char_offset);
    const keptCount = Number(parsed.kept_paragraph_count);
    const inputCount = Number(parsed.input_paragraph_count);
    return {
      status: typeof parsed.status === "string" ? parsed.status : "",
      marker: typeof parsed.marker === "string" ? parsed.marker : "",
      endIndex: Number.isFinite(endIndex) ? endIndex : null,
      endCharOffset: Number.isFinite(endCharOffset) ? endCharOffset : null,
      splitBoundary: parsed.split_boundary_paragraph === true,
      keptParagraphCount: Number.isFinite(keptCount) ? keptCount : null,
      inputParagraphCount: Number.isFinite(inputCount) ? inputCount : null,
    };
  } catch {
    return null;
  }
}

export function parseCaseSymbolProfileExport(rawJson) { return parseCaseSymbolProfile(rawJson); }

function _normalizeCaseSymbolProfile(parsed) {
  if (!parsed || typeof parsed !== "object") {
    return null;
  }

  if (Array.isArray(parsed.symbol_analyses)) {
    return parsed;
  }

  if (typeof parsed.truth_value === "string") {
    const normalized = {
      id: typeof parsed.id === "string" ? parsed.id : "",
      output_type: typeof parsed.output_type === "string" ? parsed.output_type : "",
      symbol_analyses: [
        {
          symbol_id: "constraint",
          truth_value: parsed.truth_value,
          issue_summary: typeof parsed.issue_summary === "string" ? parsed.issue_summary : "",
          facts_summary: typeof parsed.facts_summary === "string" ? parsed.facts_summary : "",
          application_summary: typeof parsed.application_summary === "string" ? parsed.application_summary : "",
          reasoning_chain: typeof parsed.reasoning_chain === "string" ? parsed.reasoning_chain : "",
          relevant_texts: Array.isArray(parsed.relevant_texts)
            ? parsed.relevant_texts.filter((txt) => typeof txt === "string")
            : [],
        },
      ],
    };
    return normalized;
  }

  return null;
}

function parseCaseSymbolProfile(rawJson) {
  if (!rawJson) {
    return null;
  }
  try {
    const parsed = typeof rawJson === "string" ? JSON.parse(rawJson) : rawJson;
    if (!parsed || typeof parsed !== "object") return null;
    if (parsed.profile && typeof parsed.profile === "object") {
      return _normalizeCaseSymbolProfile(parsed.profile);
    }
    return _normalizeCaseSymbolProfile(parsed);
  } catch {
    return null;
  }
}

/** Build a symbolId → node lookup from corpus_level_cfg.json nodes list. */
