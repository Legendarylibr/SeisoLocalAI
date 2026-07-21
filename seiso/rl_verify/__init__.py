"""Shared verifiable rewards for slime GRPO and distill-RL.

Outcome scores come from hard checks (exact, numeric, choice, sandboxed code).
Format rewards inspect the *generated* completion only — never post-hoc rewritten
strings. Code proofs run unit tests in a restricted subprocess.
"""

from __future__ import annotations

from seiso.rl_verify.code_corpus import (
    corpus_stats,
    generate_code_corpus,
    generate_grounded_task,
)
from seiso.rl_verify.code_proof import (
    CodeProofResult,
    extract_python_code,
    verify_code_proof,
)
from seiso.rl_verify.extract import (
    extract_choice,
    final_answer_text,
    format_thinking_prompt,
    has_closed_thinking_trace,
    last_number,
    normalize_answer,
    split_thinking_trace,
)
from seiso.rl_verify.preferences import (
    PreferencePair,
    ScoredCompletion,
    preference_row_from_pair,
    score_code_completion,
    select_preference_pair,
)
from seiso.rl_verify.synth_code import (
    CodeTask,
    SynthBundle,
    build_preference,
    emit_standard_artifacts,
    synthesize_code_bundle,
    validate_task,
)
from seiso.rl_verify.verify import (
    VerifierResult,
    code_outcome_value,
    format_reward,
    outcome_reward,
    resolve_checker,
    resolve_code_reward_mode,
    score_completion,
    verify_outcome,
)

__all__ = [
    "CodeProofResult",
    "CodeTask",
    "PreferencePair",
    "ScoredCompletion",
    "SynthBundle",
    "VerifierResult",
    "build_preference",
    "code_outcome_value",
    "corpus_stats",
    "emit_standard_artifacts",
    "extract_choice",
    "extract_python_code",
    "final_answer_text",
    "format_reward",
    "format_thinking_prompt",
    "generate_code_corpus",
    "generate_grounded_task",
    "has_closed_thinking_trace",
    "last_number",
    "normalize_answer",
    "outcome_reward",
    "preference_row_from_pair",
    "resolve_checker",
    "resolve_code_reward_mode",
    "score_code_completion",
    "score_completion",
    "select_preference_pair",
    "split_thinking_trace",
    "synthesize_code_bundle",
    "validate_task",
    "verify_code_proof",
    "verify_outcome",
]
