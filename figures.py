#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REG_FILE = "reg_models_accs.json"
CLF_FILE = "class_models_accs.json"
RAND_FILE = "random_class_models_accs.json"

DPI = 400
FIGSIZE = (6, 6)

def load_scores(path):
    """Read {rat: {model_group: [scores]}} into a long dataframe."""
    with open(path, "r") as f:
        data = json.load(f)

    rows = [
        {"Rat ID": rat_id, "Model": model, "Score": score}
        for rat_id, models in data.items()
        for model, scores in models.items()
        for score in scores
    ]
    return pd.DataFrame(rows)

def select_best_per_rat(df):
    """Keep only each rat's highest-mean-scoring model group.
    Returns the filtered long dataframe, the rat order (best to worst),
    and the rat to winning model mapping.
    """
    means = df.groupby(["Rat ID", "Model"])["Score"].mean().reset_index()
    best = means.loc[means.groupby("Rat ID")["Score"].idxmax()]
    winner = dict(zip(best["Rat ID"], best["Model"]))

    filtered = df[df.apply(lambda r: r["Model"] == winner[r["Rat ID"]], axis=1)]
    rat_order = (
        filtered.groupby("Rat ID")["Score"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return filtered, rat_order, winner

def build_palette(*dfs):
    """One consistent colour per model group across both figures."""
    models = sorted({m for df in dfs for m in df["Model"].unique()})
    colors = sns.color_palette("Set3", n_colors=len(models))
    return {m: colors[i] for i, m in enumerate(models)}

def plot_boxes(data, rat_order, palette, title, xlabel, stem, chance=None):
    plt.figure(figsize=FIGSIZE)
    sns.boxplot(
        data=data, y="Rat ID", x="Score", hue="Model",
        order=rat_order, palette=palette, showfliers=False, zorder=2,
    )
    if chance is not None:
        sns.scatterplot(
            data=chance, y="Rat ID", x="percentile_95",
            color="black", marker="*", s=100, alpha=0.7, zorder=3,
            label="Chance-level 95th percentile",
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Rat ID")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{stem}.svg", format="svg")
    plt.savefig(f"{stem}.png", format="png", dpi=DPI)
    plt.close()

def chance_thresholds(rand_df, winner, rat_order):
    """95th percentile of the permutation distribution, per rat, for the
    model group that won on the real data."""
    matched = rand_df[rand_df.apply(lambda r: r["Model"] == winner[r["Rat ID"]], axis=1)]
    thresholds = (
        matched.groupby("Rat ID")["Score"]
        .quantile(0.95)
        .rename("percentile_95")
        .reset_index()
    )
    # Same row order as the boxplot categories
    return thresholds.set_index("Rat ID").loc[rat_order].reset_index()

if __name__ == "__main__":
    reg_df = load_scores(REG_FILE)
    clf_df = load_scores(CLF_FILE)
    rand_df = load_scores(RAND_FILE)

    palette = build_palette(reg_df, clf_df)

    # Figure 1: time-to-event regression
    reg_best, reg_order, _ = select_best_per_rat(reg_df)
    plot_boxes(
        reg_best, reg_order, palette,
        title="Best-performing model for time-to-event regression",
        xlabel="Cross-validation R\u00b2 score",
        stem="time_to_event_reg",
    )
    # Figure 2: pre-event sequence classification
    clf_best, clf_order, clf_winner = select_best_per_rat(clf_df)
    chance = chance_thresholds(rand_df, clf_winner, clf_order)
    plot_boxes(
        clf_best, clf_order, palette,
        title="Best-performing model for pre-event sequence classification",
        xlabel="Cross-validation accuracy",
        stem="pre_event_seq_classification",
        chance=chance,
    )
