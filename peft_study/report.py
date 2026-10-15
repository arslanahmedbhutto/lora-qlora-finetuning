"""Build the results table and figures from results/runs/*.json.

    python -m peft_study.report
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from peft_study import FIGURES_DIR, RESULTS_DIR, RUNS_DIR
from peft_study.experiments import EXPERIMENTS

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_runs() -> list[dict]:
    order = [e.name for e in EXPERIMENTS]
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in RUNS_DIR.glob("*.json")]
    return sorted(runs, key=lambda r: order.index(r["name"]))


def fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def table(runs: list[dict]) -> str:
    full = next((r for r in runs if r["method"] == "full"), None)
    head = ("| Method | Trainable params | Trainable % | Test accuracy | Test macro F1 | "
            "Gap to full FT (acc) | Stored per task | Train time |\n|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in runs:
        gap = f"{100 * (r['test']['accuracy'] - full['test']['accuracy']):+.1f} pt" if full and r is not full else "—"
        rows.append(
            f"| {r['label']} | {fmt_params(r['trainable_params'])} | {r['trainable_pct']:.2f}% | "
            f"{100 * r['test']['accuracy']:.1f}% | {r['test']['macro_f1']:.3f} | {gap} | "
            f"{r['artifact_mb']:.1f} MB | {r['train_time_s'] / 60:.1f} min |"
        )
    return head + "\n".join(rows) + "\n"


def tradeoff_figure(runs: list[dict], path):
    fig, ax = plt.subplots(figsize=(9, 5))
    qv = [r for r in runs if r["method"] == "lora" and r["target_modules"] == ["q_lin", "v_lin"]]
    if qv:
        ax.plot([r["trainable_params"] for r in qv], [100 * r["test"]["accuracy"] for r in qv],
                "-o", color=BLUE, lw=2, ms=8, label="LoRA on attention q,v (r = 2, 8, 32)")
        for r in qv:
            ax.annotate(f"r={r['r']}", (r["trainable_params"], 100 * r["test"]["accuracy"]),
                        textcoords="offset points", xytext=(0, -18), ha="center", color=BLUE, fontsize=9)
    styles = {"head": (GRAY, "s", "Head only (frozen encoder)"),
              "full": (AQUA, "D", "Full fine-tuning"),
              "lora_all": (ORANGE, "^", "LoRA r=8 on all linear layers")}
    for r in runs:
        key = "lora_all" if r["method"] == "lora" and len(r["target_modules"]) > 2 else r["method"]
        if key in styles:
            c, m, lab = styles[key]
            ax.plot(r["trainable_params"], 100 * r["test"]["accuracy"], m, color=c, ms=10, label=lab,
                    markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs trainable parameters (DistilBERT, 6 emotions)")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def storage_figure(runs: list[dict], path):
    names = [r["label"] for r in runs][::-1]
    sizes = [r["artifact_mb"] for r in runs][::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(runs) + 1.2))
    colors = [AQUA if "Full" in n else GRAY if "Head" in n else BLUE for n in names]
    ax.barh(names, sizes, color=colors, height=0.6)
    for i, s in enumerate(sizes):
        ax.text(s * 1.08, i, f"{s:.1f} MB", va="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(right=max(sizes) * 4)
    ax.set_xlabel("Storage needed per task on top of the shared base model (MB, log scale)")
    ax.set_title("What you ship per task")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("No runs in results/runs/. Run `python -m peft_study.train` first.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    md = table(runs)
    (RESULTS_DIR / "results.md").write_text("# LoRA study results (test set, 2,000 texts)\n\n" + md, encoding="utf-8")
    tradeoff_figure(runs, FIGURES_DIR / "accuracy_vs_trainable_params.png")
    storage_figure(runs, FIGURES_DIR / "storage_per_task.png")
    print(md)


if __name__ == "__main__":
    main()
