"""Run the fine-tuning experiments. Each finished run is saved to results/runs/<name>.json
and skipped next time, so a long CPU run can be stopped and resumed.

    python -m peft_study.train                      # all experiments
    python -m peft_study.train --only lora_r8_qv    # one experiment
    python -m peft_study.train --train-size 1000    # quick smoke run
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from peft_study import ROOT, RUNS_DIR, SEED
from peft_study.data import LABELS, load_split, stratified_sample
from peft_study.experiments import BASE_MODEL, EXPERIMENTS, HEAD, BY_NAME, Experiment

ADAPTERS_DIR = ROOT / "adapters"
MAX_LEN = 64       # 99% of texts are under 50 tokens
EPOCHS = 3
BATCH = 32


def build_model(exp: Experiment):
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )
    if exp.method == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.split(".")[0] in HEAD
    elif exp.method == "lora":
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=exp.r, lora_alpha=2 * exp.r, lora_dropout=0.1,
            target_modules=exp.target_modules,
            # DistilBERT's pre_classifier is new and randomly initialised: it must train too
            modules_to_save=HEAD,
        ))
    return model


def count_params(model) -> dict:
    total = sum(p.numel() for n, p in model.named_parameters() if "modules_to_save" not in n
                and "original_module" not in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return {"total_params": total, "trainable_params": trainable, "lora_params": lora,
            "trainable_pct": 100 * trainable / total}


def loader(df, tok, shuffle, seed=SEED):
    enc = tok(df["text"].tolist(), truncation=True, max_length=MAX_LEN)
    items = [{"input_ids": i, "attention_mask": m, "labels": l}
             for i, m, l in zip(enc["input_ids"], enc["attention_mask"], df["label"])]

    def collate(batch):
        out = tok.pad([{k: b[k] for k in ("input_ids", "attention_mask")} for b in batch], return_tensors="pt")
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        return out

    g = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=BATCH, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def evaluate(model, dl) -> dict:
    model.eval()
    preds, gold = [], []
    for batch in dl:
        labels = batch.pop("labels")
        preds.append(model(**batch).logits.argmax(-1).numpy())
        gold.append(labels.numpy())
    p, g = np.concatenate(preds), np.concatenate(gold)
    return {
        "accuracy": float(accuracy_score(g, p)),
        "macro_f1": float(f1_score(g, p, average="macro")),
        "per_class_f1": dict(zip(LABELS, map(float, f1_score(g, p, average=None, labels=range(len(LABELS)))))),
    }


def artifact_size_mb(model, exp: Experiment, name: str) -> float:
    """Size of what you'd need to store to deploy this run on top of the shared base model."""
    if exp.method == "lora":
        out = ADAPTERS_DIR / name
        if out.exists():
            shutil.rmtree(out)
        model.save_pretrained(out)
        return sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20
    trainable = [p for p in model.parameters() if p.requires_grad]
    return sum(p.numel() * p.element_size() for p in trainable) / 2**20


def run(exp: Experiment, train_df, val_df, test_df, tok) -> dict:
    torch.manual_seed(SEED)
    model = build_model(exp)
    params = count_params(model)
    print(f"\n== {exp.label}: {params['trainable_params']:,} trainable ({params['trainable_pct']:.2f}%)", flush=True)

    train_dl = loader(train_df, tok, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=exp.lr, weight_decay=0.01)
    steps = EPOCHS * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    history, t0 = [], time.time()
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for batch in train_dl:
            loss = model(**batch).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item())
        val = evaluate(model, loader(val_df, tok, shuffle=False))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        print(f"   epoch {epoch + 1}: loss {history[-1]['train_loss']:.4f}  val acc {val['accuracy']:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    train_time = time.time() - t0

    test = evaluate(model, loader(test_df, tok, shuffle=False))
    print(f"   test acc {test['accuracy']:.4f}  macro-F1 {test['macro_f1']:.4f}", flush=True)
    return {
        "name": exp.name, "label": exp.label, "method": exp.method, "r": exp.r,
        "target_modules": exp.target_modules, "lr": exp.lr, **params,
        "artifact_mb": artifact_size_mb(model, exp, exp.name),
        "train_time_s": train_time, "history": history,
        "val": history[-1], "test": test,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--force", action="store_true", help="re-run finished experiments")
    args = ap.parse_args()

    train_df = stratified_sample(load_split("train"), args.train_size)
    val_df, test_df = load_split("val"), load_split("test")
    print(f"train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,} | device cpu", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for exp in [BY_NAME[n] for n in args.only] if args.only else EXPERIMENTS:
        out = RUNS_DIR / f"{exp.name}.json"
        if out.exists() and not args.force:
            print(f"skip {exp.name} (done)")
            continue
        result = run(exp, train_df, val_df, test_df, tok)
        result.update({"train_size": len(train_df), "epochs": EPOCHS, "batch_size": BATCH, "max_len": MAX_LEN,
                       "base_model": BASE_MODEL, "torch": torch.__version__,
                       "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
