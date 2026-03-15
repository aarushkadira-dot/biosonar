import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just saving to file
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix, classification_report

# reads test_results.csv and training_log.csv from train.py
# outputs 3 pngs: confusion matrix, f1 bar chart, learning curve
MODEL_DIR  = "model"
OUTPUT_DIR = "model"

# full name mapping - test_results.csv uses the folder names from ImageFolder
FULL_NAMES = {
    "beaked_whale":       "Beaked Whale",
    "blue_whale":         "Blue Whale",
    "false_killer_whale": "False Killer Whale",
    "fin_whale":          "Fin Whale",
    "humpback_whale":     "Humpback Whale",
    "orca":               "Orca",
    "pilot_whale":        "Pilot Whale",
    "sperm_whale":        "Sperm Whale",
}

# dark ocean color scheme - matches the marine bio theme :)
# BG is basically black navy, accents are cyan and green
BG      = "#0a0f1e"
ACCENT  = "#00d4ff"
ACCENT2 = "#00ff9d"
TEXT    = "#e8f4f8"
GRID    = "#1a2540"


def style():
    # apply dark theme so we dont have to set it on every plot
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.6,
        "font.family":       "monospace",  # monospace fits the sonar/technical vibe
        "font.size":         10,
    })


def confusion(results, classes):
    # normalized confusion matrix - each row sums to 1
    # so we can see error rates even for species with few test samples (beaked whale)
    # without normalization beaked whale's row would look fine just bc theres fewer samples

    true_mapped = results["true"].map(lambda x: FULL_NAMES.get(x, x))
    pred_mapped = results["predicted"].map(lambda x: FULL_NAMES.get(x, x))
    class_names = [FULL_NAMES[c] for c in sorted(FULL_NAMES.keys())]

    cm = confusion_matrix(true_mapped, pred_mapped, labels=class_names)

    # normalize by row (true label) so each cell = fraction of that species predicted as X
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BG)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # shorten labels so they fit on the axes - prevents overlapping
    short_labels = [
        FULL_NAMES[c].replace(" Whale", "").replace("False Killer", "Fls. Killer")
        for c in sorted(FULL_NAMES.keys())
    ]

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # annotate each cell with its value
    # flip text color so it stays readable on both light and dark cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            color = "white" if val < 0.5 else "#0a0f1e"
            # bold the diagonal (correct predictions) so they stand out
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)

    ax.set_title("Confusion Matrix — Normalized", fontsize=13, pad=15, color=ACCENT, fontweight="bold")
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("True", labelpad=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved {out}")


def f1_bars(results):
    # f1 = harmonic mean of precision and recall
    # better metric than accuracy alone bc it catches cases where model ignores rare classes
    # beaked whale (400 train samples) is the one to watch here

    true_mapped = results["true"].map(lambda x: FULL_NAMES.get(x, x))
    pred_mapped = results["predicted"].map(lambda x: FULL_NAMES.get(x, x))
    class_names = [FULL_NAMES[c] for c in sorted(FULL_NAMES.keys())]

    # output_dict=True gives us a dict we can actually work with
    report = classification_report(true_mapped, pred_mapped, labels=class_names, output_dict=True)

    species = []
    f1s     = []
    for c in class_names:
        if c in report:
            # shorten for chart readability
            species.append(c.replace(" Whale", "").replace("False Killer", "Fls. Killer"))
            f1s.append(report[c]["f1-score"])

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    # color code by performance tier:
    # green = excellent (>=0.99), cyan = good (>=0.95), red = needs attention (<0.95)
    colors = [ACCENT2 if f >= 0.99 else ACCENT if f >= 0.95 else "#ff6b6b" for f in f1s]
    bars = ax.barh(species, f1s, color=colors, height=0.55, zorder=3)

    # put the value inside the bar so it doesnt overlap with anything
    for bar, val in zip(bars, f1s):
        ax.text(val - 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="right",
                fontsize=9, color=BG, fontweight="bold")

    # zoom in so differences are visible - 0 to 1 would make everything look the same
    ax.set_xlim(0.85, 1.01)
    ax.axvline(1.0, color=GRID, linewidth=1, linestyle="--", zorder=2)  # perfect score line
    ax.set_xlabel("F1 Score", labelpad=10)
    ax.set_title("Per-Species F1 Score", fontsize=13, pad=15, color=ACCENT, fontweight="bold")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "f1_scores.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved {out}")

    # full text report for terminal - useful for devpost writeup numbers
    print("\nclassification report:")
    print(classification_report(true_mapped, pred_mapped, labels=class_names))


def learning_curve(log_path):
    # plots train vs val accuracy and loss over epochs
    # good training run = val tracks train without falling behind (no overfitting)
    # the dotted line marks when we unfroze layer3+layer4 at epoch 6

    df = pd.read_csv(log_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)

    epochs = df["epoch"]

    # --- accuracy subplot ---
    ax1.plot(epochs, df["train_acc"], color=ACCENT,  linewidth=2, label="train", zorder=3)
    ax1.plot(epochs, df["val_acc"],   color=ACCENT2, linewidth=2, label="val",   zorder=3)

    # vertical marker at epoch 5 = where backbone unfreezing happened
    # val acc jumps sharply right after this - shows the unfreeze actually helped
    ax1.axvline(5, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.4)
    ax1.text(5.5, df["train_acc"].min(), "unfreeze", fontsize=7, color="#ffffff", alpha=0.5)

    ax1.set_xlabel("Epoch", labelpad=8)
    ax1.set_ylabel("Accuracy", labelpad=8)
    ax1.set_title("Accuracy", fontsize=12, color=ACCENT, fontweight="bold")
    ax1.legend(framealpha=0.2, facecolor=BG, edgecolor=GRID)
    ax1.grid(True, zorder=0)
    ax1.set_ylim(0.6, 1.02)  # start at 0.6 so early epochs are readable

    # --- loss subplot ---
    ax2.plot(epochs, df["train_loss"], color=ACCENT,  linewidth=2, label="train", zorder=3)
    ax2.plot(epochs, df["val_loss"],   color=ACCENT2, linewidth=2, label="val",   zorder=3)
    ax2.axvline(5, color="#ffffff", linewidth=0.8, linestyle=":", alpha=0.4)

    ax2.set_xlabel("Epoch", labelpad=8)
    ax2.set_ylabel("Loss", labelpad=8)
    ax2.set_title("Loss", fontsize=12, color=ACCENT, fontweight="bold")
    ax2.legend(framealpha=0.2, facecolor=BG, edgecolor=GRID)
    ax2.grid(True, zorder=0)

    fig.suptitle("Training History — BioSonar ResNet-34", fontsize=14,
                 color=TEXT, fontweight="bold", y=1.02)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "learning_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    style()  # set dark theme before any plotting

    results_path = os.path.join(MODEL_DIR, "test_results.csv")
    log_path     = os.path.join(MODEL_DIR, "training_log.csv")

    # both files are generated by train.py - cant run this without them
    if not os.path.exists(results_path):
        print(f"missing {results_path} - run train.py first")
        return
    if not os.path.exists(log_path):
        print(f"missing {log_path} - run train.py first")
        return

    results = pd.read_csv(results_path)
    classes = results["true"].unique().tolist()

    # quick sanity check before generating plots
    test_acc = (results["true"] == results["predicted"]).mean()
    print(f"test accuracy: {test_acc:.4f} ({int(test_acc * len(results))}/{len(results)} correct)\n")

    confusion(results, classes)
    f1_bars(results)
    learning_curve(log_path)

    print("\ndone — saved confusion_matrix.png, f1_scores.png, learning_curve.png to model/")


if __name__ == "__main__":
    main()
