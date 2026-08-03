"""Tokenization and batch collation for prompt / chosen / rejected preference data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DPODataCollator:
    """Build padded batches for DPO from raw string preference fields.

    Each example must provide ``prompt``, ``chosen``, and ``rejected`` strings.
    The collator concatenates ``prompt + completion``, masks prompt tokens in
    ``labels`` with ``-100``, and pads to the longest sequence in the batch.
    """

    tokenizer: Any
    max_prompt_length: int = 512
    max_length: int = 1024
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = None
    use_chat_template: bool = False

    def _pad_token_id(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id or eos_token_id; set tokenizer.pad_token "
                "before constructing DPODataCollator."
            )
        return int(pad_token_id)

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_template is None:
            return prompt
        formatted = apply_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(formatted)

    def _tokenize_pair(self, prompt: str, completion: str) -> dict[str, Any]:
        """Tokenize ``prompt + completion`` jointly and build label mask.

        Joint encoding avoids BPE boundary artifacts from concatenating separately
        tokenized prompt/completion id lists.
        """
        prompt_text = self._format_prompt(prompt)
        add_special = not self.use_chat_template
        prompt_ids = list(
            self.tokenizer(
                prompt_text,
                add_special_tokens=add_special,
                truncation=True,
                max_length=self.max_prompt_length,
            )["input_ids"]
        )
        full_ids = list(
            self.tokenizer(
                prompt_text + completion,
                add_special_tokens=add_special,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
        )
        # Longest common prefix with prompt-only ids = supervised start boundary.
        prompt_len = 0
        for left, right in zip(prompt_ids, full_ids, strict=False):
            if left != right:
                break
            prompt_len += 1
        # Fall back when LCP is empty or only a trivial prefix (e.g. shared BOS).
        trivial_lcp = (
            prompt_ids
            and full_ids
            and (
                prompt_len == 0
                or (
                    len(prompt_ids) > 2
                    and prompt_len < max(2, int(0.5 * len(prompt_ids)))
                )
            )
        )
        if trivial_lcp:
            completion_ids = list(
                self.tokenizer(
                    completion,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(1, self.max_length - len(prompt_ids)),
                )["input_ids"]
            )
            full_ids = (prompt_ids + completion_ids)[: self.max_length]
            prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        labels[:prompt_len] = [self.label_pad_token_id] * prompt_len
        return {
            "input_ids": full_ids,
            "labels": labels,
            "prompt_length": prompt_len,
        }

    def _pad_sequences(
        self,
        sequences: list[list[int]],
        *,
        pad_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in sequences)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        padded = []
        attention_masks = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded.append(seq + [pad_value] * pad_len)
            attention_masks.append([1] * len(seq) + [0] * pad_len)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(attention_masks, dtype=torch.long),
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        chosen_input_ids: list[list[int]] = []
        chosen_labels: list[list[int]] = []
        rejected_input_ids: list[list[int]] = []
        rejected_labels: list[list[int]] = []

        pad_token_id = self._pad_token_id()

        for row in features:
            prompt = row["prompt"]
            chosen = row["chosen"]
            rejected = row["rejected"]

            chosen_pair = self._tokenize_pair(prompt, chosen)
            rejected_pair = self._tokenize_pair(prompt, rejected)

            chosen_input_ids.append(chosen_pair["input_ids"])
            chosen_labels.append(chosen_pair["labels"])
            rejected_input_ids.append(rejected_pair["input_ids"])
            rejected_labels.append(rejected_pair["labels"])

        chosen_ids, chosen_mask = self._pad_sequences(
            chosen_input_ids, pad_value=pad_token_id
        )
        chosen_lbls, _ = self._pad_sequences(
            chosen_labels, pad_value=self.label_pad_token_id
        )
        rejected_ids, rejected_mask = self._pad_sequences(
            rejected_input_ids,
            pad_value=pad_token_id,
        )
        rejected_lbls, _ = self._pad_sequences(
            rejected_labels, pad_value=self.label_pad_token_id
        )

        # Padding positions must also be ignored in the loss.
        chosen_lbls = chosen_lbls.masked_fill(chosen_mask == 0, self.label_pad_token_id)
        rejected_lbls = rejected_lbls.masked_fill(
            rejected_mask == 0, self.label_pad_token_id
        )

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "chosen_labels": chosen_lbls,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
            "rejected_labels": rejected_lbls,
        }
