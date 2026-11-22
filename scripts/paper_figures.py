"""Paper figures from docs/paper/results.json (written by scripts/paper_tables.py).

Writes PDF (vector, for LaTeX/Word) and PNG (for slides) into docs/paper/figures/:

* fig1_attack_success   attack success per model, model alone vs through the gateway,
                        one panel per judge, 95% Wilson intervals
* fig2_utility          MedQA accuracy and medical-question refusal, alone vs gateway
* fig3_stages           where the gateway stopped each attack prompt, per model
* fig4_suites           attack success per attack benchmark, pooled over models
* fig5_ablations        attack success, medical refusal and MedQA accuracy per
                        gateway configuration (written once ablations exist)

Colours follow a validated categorical order (blue, orange, aqua, yellow) with a
neutral grey for "not stopped"; every bar carries its value, so identity never
relies on colour alone.

    python scripts/paper_figures.py [--results docs/paper/results.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8f8e89", "#e7e6e2", "#ffffff"
ALONE, GATEWAY = "#eb6834", "#2a78d6"          # categorical slots 2 and 1
STAGE_COLORS = {                                # pipeline order; slots 1-4 + neutral
    "input detectors": ("4 · Attack detectors", "#2a78d6"),
    "grounding check": ("8 · Grounding check", "#eb6834"),
    "output guard": ("9 · Output guard", "#1baf7a"),
    "model refused": ("Model refused", "#eda100"),
    "answered": ("Answered (not stopped)", "#c9c8c3"),
}
SUITE_NAMES = {
    "medsafetybench": "MedSafetyBench", "medical_jailbreak": "Medical jailbreaks",
    "in_the_wild": "In-the-wild jailbreaks", "jbb_harmful": "JailbreakBench harmful",
    "deepset_injection": "deepset injections", "xstest_unsafe": "XSTest unsafe",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def _has(r) -> bool:
    return bool(r) and r.get("n") and r.get("pct") is not None


def _err(r):
    if not _has(r):
        return [[0], [0]]
    return [[max(r["pct"] - r["lo"], 0)], [max(r["hi"] - r["pct"], 0)]]


def paired_panel(ax, labels, alone, gateway, title, xmax=100, unit="%"):
    """Horizontal grouped bars: model alone (top) vs through the gateway (bottom)."""
    h = 0.36
    for i, (lab, a, g) in enumerate(zip(labels, alone, gateway)):
        y = len(labels) - 1 - i
        for off, r, color, name in ((h / 2, a, ALONE, "Model alone (undefended)"),
                                    (-h / 2, g, GATEWAY, "Through MedLM gateway (defended)")):
            if _has(r):
                ax.barh(y + off, r["pct"], height=h * 0.92, color=color, label=name if i == 0 else None,
                        xerr=_err(r), error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 2})
                ax.text(min(r["hi"], xmax) + 1.2, y + off, f"{r['pct']:.1f}{unit}", va="center",
                        fontsize=7.5, color=INK)
            else:
                ax.text(1, y + off, "n/a", va="center", fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", color=INK)
    ax.set_axisbelow(True)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def _legend(fig, ax, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.04), fontsize=8)


def fig_attack_success(main):
    models = list(main)
    panels = [("asr_j1", "Judge 1: Llama Guard 3 8B")]
    if any(_has(main[m]["undefended"].get("asr_j2")) for m in models):
        panels.append(("asr_j2", "Judge 2: Qwen3-8B rubric (different model family)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 0.6, 2.6), sharey=True, squeeze=False)
    xmax = max([main[m][c][k]["hi"] or 0 for m in models for c in ("undefended", "defended")
                for k, _ in panels if _has(main[m][c].get(k))] + [10]) + 12
    for ax, (key, title) in zip(axes[0], panels):
        paired_panel(ax, models, [main[m]["undefended"].get(key) for m in models],
                     [main[m]["defended"].get(key) for m in models], title, xmax=xmax)
        ax.set_xlabel("Attack prompts with a harmful final answer (%)")
    fig.suptitle("Attack success, model alone vs through the gateway", x=0.02, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    _legend(fig, axes[0][0])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig1_attack_success")


def fig_utility(main):
    models = list(main)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.6), sharey=True)
    paired_panel(axes[0], models, [main[m]["undefended"]["medqa_acc"] for m in models],
                 [main[m]["defended"]["medqa_acc"] for m in models], "MedQA accuracy (higher is better)", xmax=112)
    axes[0].set_xlabel("Questions answered correctly (%)")
    paired_panel(axes[1], models, [main[m]["undefended"]["med_refused"] for m in models],
                 [main[m]["defended"]["med_refused"] for m in models],
                 "Medical questions refused (lower is better)", xmax=100)
    axes[1].set_xlabel("MedQA and MedQuAD questions refused or withheld (%)")
    fig.suptitle("Clinical usefulness, model alone vs through the gateway", x=0.02, ha="left",
                 fontsize=10.5, fontweight="bold", color=INK)
    _legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig2_utility")


def fig_stages(stages):
    models = [m for m, c in stages.items() if sum(c.values())]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.8, 0.55 * len(models) + 1.2))
    for i, m in enumerate(models):
        y, left, total = len(models) - 1 - i, 0, sum(stages[m].values())
        for key, (label, color) in STAGE_COLORS.items():
            n = stages[m].get(key, 0)
            if not n:
                continue
            ax.barh(y, n, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=1.5)
            if n / total >= 0.035:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if key == "input detectors" else INK)
            left += n
    ax.set_yticks(range(len(models)), models[::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(sum(stages[m].values()) for m in models))
    ax.set_xlabel("Attack prompts sent through the gateway (count)")
    ax.set_title("Where the gateway stopped each attack prompt", loc="left", color=INK, fontsize=10.5)
    present = [k for k in STAGE_COLORS if any(stages[m].get(k) for m in models)]  # pipeline order
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[k][1]) for k in present]
    ax.legend(handles, [STAGE_COLORS[k][0] for k in present], loc="lower center", ncol=len(present),
              bbox_to_anchor=(0.5, -0.62), fontsize=7.5, handlelength=1.2)
    fig.tight_layout()
    _save(fig, "fig3_stages")


def fig_suites(suites):
    names = [s for s in SUITE_NAMES if s in suites]
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    paired_panel(ax, [SUITE_NAMES[s] for s in names], [suites[s]["undefended"]["j1"] for s in names],
                 [suites[s]["defended"]["j1"] for s in names],
                 "Attack success by attack benchmark (pooled over models, judge 1)",
                 xmax=max([suites[s]["undefended"]["j1"]["hi"] or 0 for s in names] + [10]) + 14)
    ax.set_xlabel("Attack prompts with a harmful final answer (%)")
    _legend(fig, ax)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig4_suites")


ABLATION_LABELS = {"full gateway": "Full gateway", "lg-only": "Llama Guard only", "no-rag": "No retrieval",
                   "no-input-detectors": "No input detectors", "no-llamaguard": "No Llama Guard"}


def fig_ablations(ablations):
    for model, configs in ablations.items():
        names = list(configs)
        if len(names) < 2:
            continue
        metrics = [("asr_j1", "Attack success (%)", "lower is better"),
                   ("med_refused", "Medical questions refused (%)", "lower is better"),
                   ("medqa_acc", "MedQA accuracy (%)", "higher is better")]
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 0.42 * len(names) + 1.3), sharey=True)
        for ax, (key, title, hint) in zip(axes, metrics):
            for i, n in enumerate(names):
                r, y = configs[n].get(key), len(names) - 1 - i
                color = GATEWAY if n == "full gateway" else "#86b6ef"
                if _has(r):
                    ax.barh(y, r["pct"], height=0.62, color=color, xerr=_err(r),
                            error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 2})
                    ax.text(min(r["hi"], 100) + 1.5, y, f"{r['pct']:.0f}", va="center", fontsize=7.5, color=INK)
            ax.set_xlim(0, 112)
            ax.set_title(f"{title}\n{hint}", loc="left", color=INK, fontsize=8.5)
            ax.set_yticks(range(len(names)), [ABLATION_LABELS.get(n, n) for n in names[::-1]])
            ax.tick_params(axis="y", length=0)
        fig.suptitle(f"Ablations on {model}: removing parts of the gateway", x=0.02, ha="left",
                     fontsize=10.5, fontweight="bold", color=INK)
        fig.tight_layout()
        _save(fig, "fig5_ablations")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "docs" / "paper" / "results.json")
    args = ap.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    fig_attack_success(data["main"])
    fig_utility(data["main"])
    fig_stages(data.get("stages", {}))
    fig_suites(data.get("suites", {}))
    fig_ablations(data.get("ablations", {}))


if __name__ == "__main__":
    main()
