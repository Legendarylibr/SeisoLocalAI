"""Generate teacher/student completions for DPO preference pairs."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from seiso.distill_rl.outcome import format_thinking_prompt, outcome_reward
from seiso.distill_rl.prompts import (
    RolloutPrompt,
    is_verifiable_prompt,
    prompt_to_verifier_sample,
)
from seiso.rl_verify.preferences import (
    ScoredCompletion,
    preference_row_from_pair,
    score_code_completion,
    select_preference_pair,
)


def generate_preference_rows(
    *,
    teacher_model: str,
    student_model: str,
    prompts: list[RolloutPrompt],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    use_chat_template: bool,
    teacher_revision: str | None = None,
    student_revision: str | None = None,
    trust_remote_code: bool = False,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    verifiable_outcome_rewards: bool = False,
    grpo_group_size: int = 1,
) -> list[dict[str, Any]]:
    """Generate preference rows with deterministic per-prompt seeds."""
    if verifiable_outcome_rewards and any(is_verifiable_prompt(p) for p in prompts):
        outcome_rows = generate_outcome_preference_rows(
            student_model=student_model,
            prompts=[prompt for prompt in prompts if is_verifiable_prompt(prompt)],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed + 10_000,
            use_chat_template=use_chat_template,
            revision=student_revision,
            trust_remote_code=trust_remote_code,
            require_thinking_trace=require_thinking_trace,
            thinking_instruction=thinking_instruction,
            grpo_group_size=grpo_group_size,
        )
        remaining_prompts = [
            prompt for prompt in prompts if not is_verifiable_prompt(prompt)
        ]
        if not remaining_prompts:
            return outcome_rows
        teacher_rows = generate_preference_rows(
            teacher_model=teacher_model,
            student_model=student_model,
            prompts=remaining_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            use_chat_template=use_chat_template,
            teacher_revision=teacher_revision,
            student_revision=student_revision,
            trust_remote_code=trust_remote_code,
            require_thinking_trace=require_thinking_trace,
            thinking_instruction=thinking_instruction,
            verifiable_outcome_rewards=False,
            grpo_group_size=grpo_group_size,
        )
        return outcome_rows + teacher_rows

    teacher_outputs = generate_completions(
        teacher_model,
        prompts,
        max_new_tokens,
        temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        revision=teacher_revision,
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
    )
    student_outputs = generate_completions(
        student_model,
        prompts,
        max_new_tokens,
        temperature,
        seed=seed + 10_000,
        use_chat_template=use_chat_template,
        revision=student_revision,
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
    )
    rows: list[dict[str, Any]] = []
    for prompt, chosen, rejected in zip(
        prompts, teacher_outputs, student_outputs, strict=True
    ):
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "prompt": prompt.text,
                "chosen": chosen,
                "rejected": rejected,
                "generation_seed": _prompt_seed(seed, prompt.prompt_id),
            }
        )
    return rows


def generate_outcome_preference_rows(
    *,
    student_model: str,
    prompts: list[RolloutPrompt],
    max_new_tokens: int,
    temperature: float,
    seed: int,
    use_chat_template: bool,
    revision: str | None = None,
    trust_remote_code: bool = False,
    require_thinking_trace: bool = True,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    grpo_group_size: int = 4,
    hard_negatives: bool = True,
) -> list[dict[str, Any]]:
    """Generate grouped candidates; keep only verified preference pairs.

    Code rows (with ``tests``) use unit-test pass fraction. Failed solutions
    become hard negatives when a same-group candidate passes tests.
    """
    verifiable_prompts = [prompt for prompt in prompts if is_verifiable_prompt(prompt)]
    if not verifiable_prompts:
        return []
    grouped_outputs = generate_completion_groups(
        student_model,
        verifiable_prompts,
        max_new_tokens,
        temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        revision=revision,
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        group_size=max(2, grpo_group_size),
    )
    rows: list[dict[str, Any]] = []
    for prompt, completions in zip(verifiable_prompts, grouped_outputs, strict=True):
        sample = prompt_to_verifier_sample(prompt)
        is_code = sample.get("tests") is not None or (
            (prompt.benchmark or "").lower()
            in {"code", "python", "humaneval", "mbpp", "code_exec"}
        )
        if is_code:
            scored = [
                score_code_completion(completion, sample)
                for completion in completions
            ]
            pair = select_preference_pair(
                scored,
                hard_negatives=hard_negatives,
                # Only keep pairs where chosen passes unit tests.
                require_chosen_pass=True,
            )
            if pair is None:
                continue
            rows.append(
                preference_row_from_pair(
                    prompt_id=prompt.prompt_id,
                    prompt=prompt.text,
                    pair=pair,
                    sample=sample,
                    generation_seed=_prompt_seed(seed, prompt.prompt_id),
                    group_size=len(scored),
                    group_rewards=[float(item.score) for item in scored],
                )
            )
            continue

        # Same policy as code: chosen must pass the verifier; rejected prefers
        # a hard fail (strongest incorrect) rather than arbitrary best/worst.
        scored_generic: list[ScoredCompletion] = []
        group_rewards: list[float] = []
        for completion in completions:
            reward = float(
                outcome_reward(completion, prompt.answer, benchmark=prompt.benchmark)
            )
            group_rewards.append(reward)
            scored_generic.append(
                ScoredCompletion(
                    completion=completion,
                    score=reward,
                    passed=reward > 0.5,
                    detail="outcome",
                    has_code=False,
                )
            )
        pair = select_preference_pair(
            scored_generic,
            hard_negatives=hard_negatives,
            require_chosen_pass=True,
        )
        if pair is None:
            continue
        rows.append(
            preference_row_from_pair(
                prompt_id=prompt.prompt_id,
                prompt=prompt.text,
                pair=pair,
                sample=sample,
                generation_seed=_prompt_seed(seed, prompt.prompt_id),
                group_size=len(scored_generic),
                group_rewards=group_rewards,
                reward_source="verifiable_outcome",
            )
        )
    return rows


def _prompt_seed(base_seed: int, prompt_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{prompt_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _format_generation_prompt(
    tokenizer, prompt: str, *, use_chat_template: bool
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompt


def generate_completions(
    model_path: str,
    prompts: list[RolloutPrompt],
    max_new_tokens: int,
    temperature: float,
    *,
    seed: int,
    use_chat_template: bool,
    revision: str | None = None,
    trust_remote_code: bool = False,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
) -> list[str]:
    groups = generate_completion_groups(
        model_path,
        prompts,
        max_new_tokens,
        temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        revision=revision,
        trust_remote_code=trust_remote_code,
        require_thinking_trace=require_thinking_trace,
        thinking_instruction=thinking_instruction,
        group_size=1,
    )
    return [group[0] if group else "" for group in groups]


def generate_completion_groups(
    model_path: str,
    prompts: list[RolloutPrompt],
    max_new_tokens: int,
    temperature: float,
    *,
    seed: int,
    use_chat_template: bool,
    revision: str | None = None,
    trust_remote_code: bool = False,
    require_thinking_trace: bool = False,
    thinking_instruction: str = (
        "Show your reasoning in <think>...</think>, then give the final answer."
    ),
    group_size: int = 1,
) -> list[list[str]]:
    from seiso.distill_rl.model_utils import load_causal_lm, release_causal_lm

    model, tokenizer, device = load_causal_lm(
        model_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    outputs: list[list[str]] = []
    try:
        for prompt in prompts:
            prompt_seed = _prompt_seed(seed, prompt.prompt_id)
            torch.manual_seed(prompt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prompt_seed)

            prompt_text = _format_generation_prompt(
                tokenizer,
                (
                    format_thinking_prompt(prompt.text, thinking_instruction)
                    if require_thinking_trace
                    else prompt.text
                ),
                use_chat_template=use_chat_template,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            generator = torch.Generator(device=device)
            generator.manual_seed(prompt_seed)
            with torch.inference_mode():
                generated = _generate_with_optional_generator(
                    model,
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                    generator=generator,
                    num_return_sequences=max(1, group_size),
                )
            prompt_outputs: list[str] = []
            for row in generated:
                new_tokens = row[input_len:]
                # Score and store raw generations only — do not inject synthetic tags.
                completion = tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                prompt_outputs.append(completion)
            outputs.append(prompt_outputs)
    finally:
        release_causal_lm(model)
    return outputs


def _generate_with_optional_generator(model, inputs: dict[str, Any], **kwargs):
    try:
        return model.generate(**inputs, **kwargs)
    except ValueError as exc:
        if "generator" not in str(exc):
            raise
        reduced = dict(kwargs)
        reduced.pop("generator", None)
        return model.generate(**inputs, **reduced)
