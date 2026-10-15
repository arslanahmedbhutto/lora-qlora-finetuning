"""Fine-tuning strategies compared in the study.

Every run fine-tunes DistilBERT for 6-class emotion classification with the same
data, epochs, batch size and sequence length. Only what is trainable (and the
learning rate that suits it) changes.
"""

from dataclasses import dataclass, field

BASE_MODEL = "distilbert-base-uncased"
ATTN_QV = ["q_lin", "v_lin"]
ALL_LINEAR = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
HEAD = ["pre_classifier", "classifier"]


@dataclass
class Experiment:
    name: str
    method: str                     # "head", "lora" or "full"
    lr: float
    r: int | None = None
    target_modules: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.method == "head":
            return "Head only (frozen encoder)"
        if self.method == "full":
            return "Full fine-tuning"
        where = "attention q,v" if self.target_modules == ATTN_QV else "all linear layers"
        return f"LoRA r={self.r} ({where})"


EXPERIMENTS = [
    Experiment("head_only", "head", lr=1e-3),
    *[Experiment(f"lora_r{r}_qv", "lora", lr=5e-4, r=r, target_modules=ATTN_QV) for r in (2, 8, 32)],
    Experiment("lora_r8_all", "lora", lr=5e-4, r=8, target_modules=ALL_LINEAR),
    Experiment("full_ft", "full", lr=5e-5),
]
BY_NAME = {e.name: e for e in EXPERIMENTS}
