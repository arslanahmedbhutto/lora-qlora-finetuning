"""Classify text with a saved LoRA adapter on top of the shared base model.

    python -m peft_study.predict "i feel so scared walking home alone"
    python -m peft_study.predict --adapter lora_r16_qv "what a lovely surprise"
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft_study import ROOT
from peft_study.data import LABELS
from peft_study.experiments import BASE_MODEL

DEFAULT_ADAPTER = "lora_r8_qv"


def load(adapter: str = DEFAULT_ADAPTER):
    base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(LABELS))
    model = PeftModel.from_pretrained(base, ROOT / "adapters" / adapter).eval()
    return AutoTokenizer.from_pretrained(BASE_MODEL), model


@torch.no_grad()
def predict(texts: list[str], tok, model) -> list[dict[str, float]]:
    enc = tok(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    probs = model(**enc).logits.softmax(-1).numpy()
    return [{LABELS[i]: float(p) for i, p in enumerate(row)} for row in probs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = ap.parse_args()
    tok, model = load(args.adapter)
    for text, probs in zip(args.text, predict(args.text, tok, model)):
        top = max(probs, key=probs.get)
        print(f"{top:<9} {probs[top]:.1%}  {text}")


if __name__ == "__main__":
    main()
