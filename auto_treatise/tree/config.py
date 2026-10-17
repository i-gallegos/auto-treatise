"""Tree-induction limits, axis types and batch sizes."""

from __future__ import annotations

import re

MAX_RECURSION_DEPTH = 3
TEST_MAX_RECURSION_DEPTH = 2

AXIS_TYPES: list[str] = [
    "interpretive_method",
    "authority_allocation",
    "constraint_predicate",
    "constraint_scope",
    "evidentiary_threshold",
    "procedural_posture",
    "engagement_depth",
    "other",
]

# How many previously created labels are shown to the model.
MAX_USED_LABELS_SHOWN = 24

# Full case issue briefs per assignment request. 
ASSIGNMENT_BATCH_SIZE = 50

# Without issue briefs, the pooled text of the first call is capped at this many
# characters, and assignment text is cut per case and chunked by characters.
RAW_STAGE1_CHAR_BUDGET = 180_000
RAW_ASSIGNMENT_TEXT_CHARS = 6_000
RAW_ASSIGNMENT_CHUNK_CHARS = 48_000

# Batched rounds spent assigning each depth's cases (one extra recovers cases
# the model omitted).
MAX_ASSIGNMENT_ROUNDS = 2

# A split whose smaller child holds under MIN_CHILD_FRACTION of the cluster is
# rejected, for clusters of at least MIN_BALANCE_CHECK_SIZE cases. The model is
# re-prompted up to MAX_AXIS_RETRIES more times, then the node stays a leaf.
MIN_CHILD_FRACTION = 0.1
MIN_BALANCE_CHECK_SIZE = 20
MAX_AXIS_RETRIES = 3

# No axis is proposed for a cluster smaller than this.
MIN_SPLIT_SIZE = 15

ROOT_ID = "n0"

_SPLIT_COUNT_RE = re.compile(r"Proposed split was (\d+)/(\d+)")

# Seed for the deterministic shuffle of stage-1 summaries.
AXIS_SHUFFLE_SEED = 42
