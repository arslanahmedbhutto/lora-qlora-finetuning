import json

import pytest

from peft_study import data, experiments, report, train
from peft_study.data import LABELS, load_split, stratified_sample
from peft_study.experiments import EXPERIMENTS, Experiment

TINY = "hf-internal-testing/tiny-random-DistilBertForSequenceClassification"


def test_splits_load_with_expected_sizes():
    for split, n in [("train", 16000), ("val", 2000), ("test", 2000)]:
        df = load_split(split)
        assert len(df) == n
        assert set(df["label"]) == set(range(len(LABELS)))


def test_stratified_sample_keeps_class_balance():
    df = load_split("train")
    s = stratified_sample(df, 4000)
    assert abs(len(s) - 4000) <= len(LABELS)
    full = df["label"].value_counts(normalize=True)
    sub = s["label"].value_counts(normalize=True)
    assert (full - sub).abs().max() < 0.005


def test_experiment_names_unique_and_labels_readable():
    names = [e.name for e in EXPERIMENTS]
    assert len(names) == len(set(names))
    assert Experiment("x", "lora", 1e-3, r=8, target_modules=experiments.ATTN_QV).label == "LoRA r=8 (attention q,v)"


@pytest.fixture
def tiny_base(monkeypatch):
    monkeypatch.setattr(train, "BASE_MODEL", TINY)


def test_head_only_trains_just_the_head(tiny_base):
    model = train.build_model(Experiment("h", "head", 1e-3))
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"pre_classifier", "classifier"}


def test_lora_trains_adapters_and_head_only(tiny_base):
    model = train.build_model(Experiment("l", "lora", 1e-3, r=4, target_modules=experiments.ATTN_QV))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in names) and any("lora_B" in n for n in names)
    assert all("lora_" in n or "modules_to_save" in n for n in names)
    counts = train.count_params(model)
    assert 0 < counts["lora_params"] < counts["trainable_params"] < counts["total_params"]


def test_higher_rank_means_more_lora_params(tiny_base):
    p = [train.count_params(train.build_model(Experiment("l", "lora", 1e-3, r=r, target_modules=experiments.ATTN_QV)))
         ["lora_params"] for r in (2, 4, 8)]
    assert p[0] < p[1] < p[2] and p[1] == 2 * p[0]


def test_one_training_run_end_to_end(tiny_base, monkeypatch, tmp_path):
    from transformers import AutoTokenizer

    monkeypatch.setattr(train, "EPOCHS", 1)
    monkeypatch.setattr(train, "ADAPTERS_DIR", tmp_path)
    tok = AutoTokenizer.from_pretrained(TINY)
    df = stratified_sample(load_split("val"), 64)
    result = train.run(Experiment("t", "lora", 1e-3, r=2, target_modules=experiments.ATTN_QV), df, df, df, tok)
    assert 0 <= result["test"]["accuracy"] <= 1
    assert result["artifact_mb"] > 0 and (tmp_path / "t" / "adapter_config.json").exists()
    assert len(result["history"]) == 1


def test_report_table_and_figures(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    for e, acc in [(EXPERIMENTS[0], 0.6), (EXPERIMENTS[3], 0.85), (EXPERIMENTS[-1], 0.9)]:
        (runs_dir / f"{e.name}.json").write_text(json.dumps({
            "name": e.name, "label": e.label, "method": e.method, "r": e.r, "target_modules": e.target_modules,
            "trainable_params": 1000 if e.method != "full" else 66_000_000, "trainable_pct": 1.0,
            "artifact_mb": 2.0, "train_time_s": 60, "test": {"accuracy": acc, "macro_f1": acc - 0.05},
        }))
    monkeypatch.setattr(report, "RUNS_DIR", runs_dir)
    runs = report.load_runs()
    assert [r["method"] for r in runs] == ["head", "lora", "full"]
    md = report.table(runs)
    assert "-30.0 pt" in md and "-5.0 pt" in md
    report.tradeoff_figure(runs, tmp_path / "a.png")
    report.storage_figure(runs, tmp_path / "b.png")
    assert (tmp_path / "a.png").stat().st_size > 0
