"""Plotting functions ported from the notebook. Nothing is plotted on import.

Every ``plot_*`` function takes the data it needs explicitly, optionally saves a PNG into ``save_dir``
and either shows the figure (``show=True``) or closes it. Colour-blind-safe palette, fixed order.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import confusion_matrix

from .config import CLASS_NAMES, NUM_CLASSES, RANDOM_REF_NAME, Config

OKABE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
KIND_COLORS = {"concept": "#0072B2", "shuffled_labels": "#8C8C8C", "random_reference": "#C8C8C8"}
KIND_LABELS = {"concept": "true concept CAV", "shuffled_labels": "random: shuffled labels",
               "random_reference": "random: reference vs reference"}


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5, "grid.alpha": 0.5,
                                         "axes.spines.top": False, "axes.spines.right": False})


def concept_colors(concepts: Sequence[str]) -> dict:
    """Each concept keeps the same colour in every plot."""
    return {c: OKABE[i % len(OKABE)] for i, c in enumerate(concepts)}


def _finish(fig, name: Optional[str], save_dir: Optional[Path], show: bool, tight: bool = True) -> None:
    if save_dir is not None and name is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / name, dpi=200, bbox_inches="tight" if tight else None)
    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------- data exploration
def show_grid(paths, titles, ncols=5, figsize_per=2.4, suptitle=None, save_dir=None, name=None, show=False):
    n = len(paths)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, p, t in zip(axes, paths, titles):
        ax.imshow(Image.open(p).convert("RGB"))
        ax.set_title(t, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, y=1.0)
    fig.tight_layout()
    _finish(fig, name, save_dir, show)


def plot_class_balance(train_labels, val_labels, test_labels, save_dir=None, name=None, show=False):
    counts = pd.DataFrame({
        "split": ["train"] * len(train_labels) + ["val"] * len(val_labels) + ["test"] * len(test_labels),
        "class": [CLASS_NAMES[l] for l in np.concatenate([train_labels, val_labels, test_labels])]})
    fig = plt.figure(figsize=(9, 3.5))
    sns.countplot(data=counts, x="class", hue="split")
    plt.xticks(rotation=40, ha="right")
    plt.title("ImageNette: images per class and split")
    fig.tight_layout()
    _finish(fig, name, save_dir, show)


# --------------------------------------------------------------------------- model evaluation
def plot_training_curves(history: dict, save_dir=None, name="training_curves.png", show=False):
    ep = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(ep, history["train_loss"], color=OKABE[0], lw=2, marker="o", ms=4, label="train")
    axes[0].plot(ep, history["val_loss"], color=OKABE[1], lw=2, marker="o", ms=4, label="validation")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    axes[0].legend()
    axes[1].plot(ep, history["train_acc"], color=OKABE[0], lw=2, marker="o", ms=4, label="train")
    axes[1].plot(ep, history["val_acc"], color=OKABE[1], lw=2, marker="o", ms=4, label="validation")
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axes[1].legend()
    fig.tight_layout()
    _finish(fig, name, save_dir, show, tight=False)


def plot_confusion_matrix(y_true, y_pred, save_dir=None, name="confusion_matrix.png", show=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig = plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("Test confusion matrix")
    plt.xticks(rotation=40, ha="right")
    fig.tight_layout()
    _finish(fig, name, save_dir, show, tight=False)


# --------------------------------------------------------------------------- TCAV results
def plot_cav_accuracy(cav_df: pd.DataFrame, cfg: Config, save_dir=None, name="cav_validation_accuracy.png",
                      show=False):
    d = cav_df.copy()
    d["concept"] = d["concept"].replace({RANDOM_REF_NAME: "random-ref (shared)"})
    n = len(cfg.tcav_layers)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4), sharey=True, squeeze=False)
    for ax, layer in zip(axes[0], cfg.tcav_layers):
        sns.boxplot(data=d[d.layer == layer], x="concept", y="val_acc", hue="kind",
                    hue_order=["concept"] + cfg.controls, palette=KIND_COLORS, ax=ax, linewidth=1, fliersize=2)
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.axhline(cfg.cav_min_acc, color="#D55E00", lw=0.8, ls="--")
        ax.set(title=f"CAV validation accuracy ({layer})", xlabel="", ylabel="validation accuracy", ylim=(0.3, 1.02))
        ax.tick_params(axis="x", rotation=25)
        h, _ = ax.get_legend_handles_labels()
        ax.legend(h, [KIND_LABELS[k] for k in ["concept"] + cfg.controls], fontsize=8, loc="lower left")
    fig.tight_layout()
    _finish(fig, name, save_dir, show)


def concept_class_matrix(res_df: pd.DataFrame, cfg: Config, layer: str, value: str) -> pd.DataFrame:
    d = res_df[(res_df.layer == layer) & (res_df.kind == "concept")]
    m = d.pivot_table(index="concept", columns="class_idx", values=value, aggfunc="mean").reindex(cfg.concepts)
    return m.rename(columns=dict(enumerate(CLASS_NAMES)))


def plot_heatmaps(res_df: pd.DataFrame, cfg: Config, layer: str, save_dir=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 3.6))
    for ax, value, cmap, title, kw in [
            (axes[0], "tcav", "Blues", "mean TCAV score", dict(vmin=0, vmax=1)),
            (axes[1], "mean_S", "RdBu_r", "mean directional derivative", dict(center=0))]:
        m = concept_class_matrix(res_df, cfg, layer, value)
        sns.heatmap(m, annot=True, fmt=".2f" if value == "tcav" else ".2g", cmap=cmap, ax=ax, cbar=True, **kw)
        for c, kname in cfg.concept_class_hypotheses.items():     # outline pre-specified hypotheses
            ax.add_patch(Rectangle((CLASS_NAMES.index(kname), cfg.concepts.index(c)), 1, 1,
                                   fill=False, edgecolor="black", lw=2))
        ax.set_title(f"{title} - {layer} (black box = hypothesis)")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    _finish(fig, f"heatmap_{layer}.png", save_dir, show)


def plot_hypothesis_comparison(res_df: pd.DataFrame, stats_confirmatory: pd.DataFrame, cfg: Config, layer: str,
                               save_dir=None, show=False):
    """Actual vs random CAVs per hypothesis pair (mean +- std, individual runs as dots)."""
    kinds = ["concept"] + cfg.controls
    pairs = list(cfg.concept_class_hypotheses.items())
    fig, ax = plt.subplots(figsize=(1.9 * len(pairs) + 3, 4.2))
    width = 0.8 / len(kinds)
    rng_j = np.random.default_rng(0)
    for i, (c, kname) in enumerate(pairs):
        k = CLASS_NAMES.index(kname)
        for j, kind in enumerate(kinds):
            conc = RANDOM_REF_NAME if kind == "random_reference" else c
            vals = res_df[(res_df.layer == layer) & (res_df.kind == kind) & (res_df.concept == conc) &
                          (res_df.class_idx == k)]["tcav"].values
            if len(vals) == 0:
                continue
            xpos = i + (j - (len(kinds) - 1) / 2) * width
            ax.bar(xpos, vals.mean(), width * 0.9, color=KIND_COLORS[kind], alpha=0.85,
                   label=KIND_LABELS[kind] if i == 0 else None)
            ax.errorbar(xpos, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, color="k",
                        capsize=3, lw=1)
            ax.scatter(xpos + rng_j.uniform(-width / 3, width / 3, len(vals)), vals, s=6, color="k",
                       alpha=0.25, zorder=3)
        s = stats_confirmatory[(stats_confirmatory.layer == layer) & (stats_confirmatory.concept == c) &
                               (stats_confirmatory.class_idx == k) & (stats_confirmatory.control == "shuffled_labels")]
        if len(s) and bool(s["supported_holm"].iloc[0]):
            ax.text(i, 1.06, "*", ha="center", fontsize=16)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{c}\n-> {k}" for c, k in pairs], fontsize=9)
    ax.set(ylim=(0, 1.15), ylabel="TCAV score (mean +- std over CAV runs)",
           title=f"True concept CAV vs random CAVs - {layer}\n"
                 f"(* = Holm-adjusted p<{cfg.alpha}, MWU vs shuffled-label control; confirmatory family)")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(kinds))
    fig.tight_layout()
    _finish(fig, f"hypothesis_vs_random_{layer}.png", save_dir, show)


def summarize_tcav(res_df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / count of TCAV scores per (layer, kind, concept, class)."""
    return res_df.groupby(["layer", "kind", "concept", "class_idx"])["tcav"].agg(["mean", "std", "count"]).reset_index()


def _bars(ax, labels, get, kinds):
    width = 0.8 / len(kinds)
    for j, kind in enumerate(kinds):
        means, errs = [], []
        for lab in labels:
            r = get(kind, lab)
            means.append(r["mean"].iloc[0] if len(r) else np.nan)
            errs.append(r["std"].iloc[0] if len(r) else np.nan)
        x = np.arange(len(labels)) + (j - (len(kinds) - 1) / 2) * width
        ax.bar(x, means, width * 0.9, yerr=np.nan_to_num(errs), color=KIND_COLORS[kind], alpha=0.9, capsize=2,
               error_kw=dict(lw=0.8), label=KIND_LABELS[kind])
    ax.set_xticks(range(len(labels)))
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")


def plot_concepts_for_class(summ: pd.DataFrame, cfg: Config, layer: str, class_name: str, save_dir=None, show=False):
    """All concepts for one target class (``summ`` from :func:`summarize_tcav`)."""
    k = CLASS_NAMES.index(class_name)
    kinds = ["concept"] + cfg.controls
    fig, ax = plt.subplots(figsize=(8, 3.6))

    def get(kind, c):
        conc = RANDOM_REF_NAME if kind == "random_reference" else c
        return summ[(summ.layer == layer) & (summ.kind == kind) & (summ.concept == conc) & (summ.class_idx == k)]

    _bars(ax, cfg.concepts, get, kinds)
    ax.set_xticklabels(cfg.concepts)
    ax.set(ylabel="TCAV score", title=f"Concepts for target class '{class_name}' - {layer}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _finish(fig, f"concepts_for_{class_name.replace(' ', '_')}_{layer}.png", save_dir, show)


def plot_concept_across_classes(summ: pd.DataFrame, cfg: Config, layer: str, concept: str, save_dir=None,
                                show=False):
    """One concept across all target classes."""
    kinds = ["concept"] + cfg.controls
    fig, ax = plt.subplots(figsize=(11, 3.6))

    def get(kind, kname):
        k = CLASS_NAMES.index(kname)
        conc = RANDOM_REF_NAME if kind == "random_reference" else concept
        return summ[(summ.layer == layer) & (summ.kind == kind) & (summ.concept == conc) & (summ.class_idx == k)]

    _bars(ax, CLASS_NAMES, get, kinds)
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right")
    ax.set(ylabel="TCAV score", title=f"Concept '{concept}' across target classes - {layer}")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    _finish(fig, f"concept_{concept}_across_classes_{layer}.png", save_dir, show)


def plot_all_tcav_results(cav_df, res_df, stats_confirmatory, cfg: Config, save_dir, show=False) -> None:
    """Every TCAV result plot of the notebook (sections 24)."""
    set_plot_theme()
    plot_cav_accuracy(cav_df, cfg, save_dir, show=show)
    for layer in cfg.tcav_layers:
        plot_heatmaps(res_df, cfg, layer, save_dir, show)
    for layer in cfg.tcav_layers:
        plot_hypothesis_comparison(res_df, stats_confirmatory, cfg, layer, save_dir, show)
    summ = summarize_tcav(res_df)
    for kname in dict.fromkeys(cfg.concept_class_hypotheses.values()):
        plot_concepts_for_class(summ, cfg, cfg.primary_layer, kname, save_dir, show)
    for c in cfg.concepts:
        plot_concept_across_classes(summ, cfg, cfg.primary_layer, c, save_dir, show)
