"""
Programmatic binary dendrogram induction over case-level constraint profiles.

Pass 1 (cases.py): one API call per opinion -> case symbol profile.
Pass 2 (this package): recursive binary case splitting for a single constraint.
"""

from .config import ASSIGNMENT_BATCH_SIZE
from .config import AXIS_SHUFFLE_SEED
from .config import AXIS_TYPES
from .config import MAX_ASSIGNMENT_ROUNDS
from .config import MAX_AXIS_RETRIES
from .config import MAX_RECURSION_DEPTH
from .config import MAX_USED_LABELS_SHOWN
from .config import MIN_BALANCE_CHECK_SIZE
from .config import MIN_CHILD_FRACTION
from .config import MIN_SPLIT_SIZE
from .config import RAW_ASSIGNMENT_CHUNK_CHARS
from .config import RAW_ASSIGNMENT_TEXT_CHARS
from .config import RAW_STAGE1_CHAR_BUDGET
from .config import ROOT_ID
from .config import TEST_MAX_RECURSION_DEPTH
from .expand import induce_grammar
from .pipeline import merge_missing_case_profiles
from .pipeline import partition_for_induction
from .pipeline import run_pipeline
from .prompts import FORMALIZATION_CORPUS_LEVEL

__all__ = [
    "MAX_RECURSION_DEPTH",
    "TEST_MAX_RECURSION_DEPTH",
    "AXIS_TYPES",
    "MAX_USED_LABELS_SHOWN",
    "ASSIGNMENT_BATCH_SIZE",
    "RAW_STAGE1_CHAR_BUDGET",
    "RAW_ASSIGNMENT_TEXT_CHARS",
    "RAW_ASSIGNMENT_CHUNK_CHARS",
    "MAX_ASSIGNMENT_ROUNDS",
    "MIN_CHILD_FRACTION",
    "MIN_BALANCE_CHECK_SIZE",
    "MAX_AXIS_RETRIES",
    "MIN_SPLIT_SIZE",
    "ROOT_ID",
    "AXIS_SHUFFLE_SEED",
    "FORMALIZATION_CORPUS_LEVEL",
    "merge_missing_case_profiles",
    "induce_grammar",
    "partition_for_induction",
    "run_pipeline",
]
