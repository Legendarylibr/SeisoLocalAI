"""Generate teacher/student completions for DPO preference pairs."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from seiso.distill_rl.prompts import RolloutPrompt


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
) -> list[dict[str, Any]]:
    """Generate preference rows with deterministic per-prompt seeds."""
    teacher_outputs = generate_completions(
        teacher_model,
        prompts,
        max_new_tokens,
        temperature,
        seed=seed,
        use_chat_template=use_chat_template,
        revision=teacher_revision,
    )
    student_outputs = generate_completions(
        student_model,
        prompts,
        max_new_tokens,
        temperature,
        seed=seed + 10_000,
        use_chat_template=use_chat_template,
        revision=student_revision,
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
) -> list[str]:
    from seiso.distill_rl.model_utils import load_causal_lm, release_causal_lm

    model, tokenizer, device = load_causal_lm(model_path, revision=revision)
    outputs: list[str] = []
    try:
        for prompt in prompts:
            prompt_seed = _prompt_seed(seed, prompt.prompt_id)
            torch.manual_seed(prompt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prompt_seed)

            prompt_text = _format_generation_prompt(
                tokenizer,
                prompt.text,
                use_chat_template=use_chat_template,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_len = int(inputs["input_ids"].shape[-1])
            generator = torch.Generator(device=device)
            generator.manual_seed(prompt_seed)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=tokenizer.pad_token_id,
                    generator=generator,
                )
            new_tokens = generated[0, input_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            outputs.append(completion)
    finally:
        release_causal_lm(model)
    return outputs
