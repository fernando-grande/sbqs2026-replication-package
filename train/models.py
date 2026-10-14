"""Transformer classifiers matching the SBQS/TCC protocol."""
import torch
import torch.nn as nn
from transformers import AutoModel


def load_encoder(model_name: str, revision: str | None, attn_implementation: str | None, gradient_checkpointing: bool):
    """Pinned checkpoint, fixed attention implementation, optional checkpointing.

    Gradient checkpointing recomputes activations in the backward pass: it is
    mathematically equivalent (not necessarily bit-identical, because the
    recomputation may reorder operations) and, when enabled for a model, it is
    enabled for every run of that model.
    """
    kwargs = {"revision": revision}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    encoder = AutoModel.from_pretrained(model_name, **kwargs)
    if gradient_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return encoder


class SingleTaskClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_classes: int,
        dropout: float = 0.1,
        revision: str | None = None,
        attn_implementation: str | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = load_encoder(model_name, revision, attn_implementation, gradient_checkpointing)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.head(self.dropout(pooled))


class MultiHeadClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        section_classes: dict[str, list[str]],
        dropout: float = 0.1,
        revision: str | None = None,
        attn_implementation: str | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = load_encoder(model_name, revision, attn_implementation, gradient_checkpointing)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict(
            {task: nn.Linear(hidden, len(labels)) for task, labels in section_classes.items()}
        )
        self.task_index = {task: i for i, task in enumerate(section_classes)}
        self.log_vars = nn.Parameter(torch.zeros(len(section_classes)))

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(outputs.last_hidden_state[:, 0])

    def forward(self, input_ids, attention_mask, task: str):
        pooled = self.encode(input_ids, attention_mask)
        return self.classifiers[task](pooled)

    def uncertainty_loss(self, task: str, loss: torch.Tensor) -> torch.Tensor:
        """Kendall homoscedastic uncertainty: 0.5 * exp(-s) L + 0.5 s, s = log sigma^2."""
        log_var = self.log_vars[self.task_index[task]]
        return 0.5 * torch.exp(-log_var) * loss + 0.5 * log_var
