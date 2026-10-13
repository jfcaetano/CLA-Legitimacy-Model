#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:20:06 2026

@author: jfcaetano
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"


files = {
    "Conditional consolidation": "scenario1.csv",
    "High fragility": "scenario2.csv",
    "Baseline": "scenario3.csv",
}

smooth_window = 1001
plot_every = 1000
fallback_threshold = 0.002
recovery_epsilon = 0.0

def smooth_array(arr, window=1):
    s = pd.Series(arr)
    if window <= 1:
        return s.to_numpy()
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


all_rows = []

for scenario_label, file in files.items():
    df = pd.read_csv(file)

    for sim_id, sim_df in df.groupby("Simulation"):
        sim_df = sim_df.sort_values("Timestep").copy()

        t_real = sim_df["Timestep"].to_numpy()
        C = smooth_array(sim_df["Method_C"], smooth_window)

        t_real = t_real[::plot_every]
        C = C[::plot_every]

        dC = np.diff(C, prepend=C[0])
        shock_idx = np.where(dC < -fallback_threshold)[0]

        recovery_times = []

        for idx in shock_idx:
            if idx == 0:
                continue

            pre_drop_level = C[idx - 1]
            shock_time = t_real[idx]

            recovered_at = np.nan
            for j in range(idx + 1, len(C)):
                if C[j] >= pre_drop_level + recovery_epsilon:
                    recovered_at = t_real[j]
                    break

            if not np.isnan(recovered_at):
                recovery_times.append(recovered_at - shock_time)

        all_rows.append({
            "scenario": scenario_label,
            "simulation_id": sim_id,
            "n_shocks": len(shock_idx),
            "mean_recovery_time": np.mean(recovery_times) if recovery_times else np.nan
        })

all_df = pd.DataFrame(all_rows)


summary_df = (
    all_df.groupby("scenario", as_index=False)
    .agg(
        mean_total_shocks=("n_shocks", "mean"),
        sd_total_shocks=("n_shocks", "std"),
        mean_recovery_time=("mean_recovery_time", "mean"),
        sd_recovery_time=("mean_recovery_time", "std"),
        n_simulations=("simulation_id", "nunique")
    )
)

summary_df = summary_df.fillna(0)


scenario_order = ["Conditional consolidation", "High fragility", "Baseline"]
summary_df["scenario"] = pd.Categorical(
    summary_df["scenario"],
    categories=scenario_order,
    ordered=True
)
summary_df = summary_df.sort_values("scenario").reset_index(drop=True)

x = np.arange(len(summary_df))
bar_width = 0.34

mean_shocks = summary_df["mean_total_shocks"].to_numpy()
sd_shocks = summary_df["sd_total_shocks"].to_numpy()

mean_recovery = summary_df["mean_recovery_time"].to_numpy()
sd_recovery = summary_df["sd_recovery_time"].to_numpy()


fig, ax1 = plt.subplots(figsize=(10.8, 6))

bars1 = ax1.bar(
    x - bar_width/2,
    mean_shocks,
    width=bar_width,
    color="orange",
    edgecolor="black",
    linewidth=1.0,
    yerr=sd_shocks,
    capsize=4,
    label="Mean total shocks"
)

ax1.set_xlabel("Scenario", fontsize=13)
ax1.set_ylabel("Mean total shocks", color="black", fontsize=13)
ax1.tick_params(axis="y", colors="black")
ax1.spines["left"].set_color("black")
ax1.spines["bottom"].set_color("black")
ax1.spines["top"].set_color("black")
ax1.set_xticks(x)
ax1.set_xticklabels(summary_df["scenario"], rotation=15)
ax1.grid(True, axis="y", alpha=0.25)

ax2 = ax1.twinx()
bars2 = ax2.bar(
    x + bar_width/2,
    mean_recovery,
    width=bar_width,
    color="steelblue",
    edgecolor="black",
    linewidth=1.0,
    yerr=sd_recovery,
    capsize=4,
    label="Mean recovery time"
)

ax2.set_ylabel("Mean recovery time (real timesteps)", color="black", fontsize=12)
ax2.tick_params(axis="y", colors="black")
ax2.spines["right"].set_color("black")
ax2.spines["top"].set_color("black")

offset1 = max(mean_shocks + sd_shocks) * 0.03 if len(mean_shocks) else 0.1
offset2 = max(mean_recovery + sd_recovery) * 0.03 if len(mean_recovery) else 0.1

for i, bar in enumerate(bars1):
    h = bar.get_height()
    y_text = h + sd_shocks[i] + offset1
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        y_text,
        f"{h:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black"
    )

for i, bar in enumerate(bars2):
    h = bar.get_height()
    y_text = h + sd_recovery[i] + offset2
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        y_text,
        f"{h:,.0f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="black"
    )

ax1.set_ylim(0, max(mean_shocks + sd_shocks) * 1.1 if len(mean_shocks) else 1)
ax2.set_ylim(0, max(mean_recovery + sd_recovery) * 1.1 if len(mean_recovery) else 1)

ax1.tick_params(axis="y", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax1.tick_params(axis="x", labelsize=12)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    ncol=2,
    frameon=False,
    fontsize=12
)

plt.title("Mean total shocks and mean recovery time across all simulation IDs", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('time_shock_chart.png', dpi=900, bbox_inches='tight')
plt.show()

print(summary_df)
print(all_df.head())
############### IMAGE 2 ####################



files = {
    "Conditional consolidation": "scenario1.csv",
    "High fragility": "scenario2.csv",
    "Baseline": "scenario3.csv",
}

smooth_window = 1001
plot_every = 1000
epsilon = 0.05  


def smooth_array(arr, window=1):
    s = pd.Series(arr)
    if window <= 1:
        return s.to_numpy()
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()

def classify_regime(gap, eps=0.05):
    """
    0 = Method B dominant
    1 = Contested / near parity
    2 = Method C dominant
    """
    regime = np.zeros_like(gap, dtype=int)
    regime[gap < -eps] = 0
    regime[np.abs(gap) <= eps] = 1
    regime[gap > eps] = 2
    return regime

def human_format(x, pos=None):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


all_rows = []

for label, file in files.items():
    df = pd.read_csv(file)

    for sim_id, sim_df in df.groupby("Simulation"):
        sim_df = sim_df.sort_values("Timestep").copy()

        t = sim_df["Timestep"].to_numpy()[::plot_every]
        B = smooth_array(sim_df["Method_B"], smooth_window)[::plot_every]
        C = smooth_array(sim_df["Method_C"], smooth_window)[::plot_every]

        gap = C - B
        regime = classify_regime(gap, epsilon)

        if len(t) > 1:
            dt = np.median(np.diff(t))
        else:
            dt = 1

        time_B = np.sum(regime == 0) * dt
        time_T = np.sum(regime == 1) * dt
        time_C = np.sum(regime == 2) * dt
        total = time_B + time_T + time_C

        all_rows.append({
            "scenario": label,
            "simulation_id": sim_id,
            "time_B_dominant": time_B,
            "time_contested": time_T,
            "time_C_dominant": time_C,
            "total": total,
            "pct_B": 100 * time_B / total if total > 0 else 0,
            "pct_T": 100 * time_T / total if total > 0 else 0,
            "pct_C": 100 * time_C / total if total > 0 else 0,
        })

all_df = pd.DataFrame(all_rows)

summary_df = (
    all_df.groupby("scenario", as_index=False)
    .agg(
        time_B_dominant=("time_B_dominant", "mean"),
        time_contested=("time_contested", "mean"),
        time_C_dominant=("time_C_dominant", "mean"),
        total=("total", "mean"),
        pct_B=("pct_B", "mean"),
        pct_T=("pct_T", "mean"),
        pct_C=("pct_C", "mean"),
        n_simulations=("simulation_id", "nunique")
    )
)

summary_df = summary_df.sort_values("total").reset_index(drop=True)


y = np.arange(len(summary_df))

b = summary_df["time_B_dominant"].to_numpy()
t = summary_df["time_contested"].to_numpy()
c = summary_df["time_C_dominant"].to_numpy()
totals = summary_df["total"].to_numpy()

pct_b = summary_df["pct_B"].to_numpy()
pct_t = summary_df["pct_T"].to_numpy()
pct_c = summary_df["pct_C"].to_numpy()

max_total = totals.max()

color_b = "darkorange"
color_t = "lightgreen"
color_c = "dodgerblue"


fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    figsize=(11, 8.5),
    sharey=True
)

ax1.barh(y, b, color=color_b, edgecolor="black", linewidth=0.8, label="Method B dominant")
ax1.barh(y, t, left=b, color=color_t, edgecolor="black", linewidth=0.8, label="Contested / near parity")
ax1.barh(y, c, left=b + t, color=color_c, edgecolor="black", linewidth=0.8, label="Method C dominant")

ax1.set_yticks(y)
ax1.set_yticklabels(summary_df["scenario"])
ax1.set_xlabel("Mean time spent across simulations (real timesteps)", fontsize=13)
ax1.set_title("Absolute time by regime", fontsize=13)
ax1.xaxis.set_major_formatter(FuncFormatter(human_format))
ax1.grid(True, axis="x", alpha=0.25)

ax1.tick_params(axis="y", labelsize=13)
ax1.tick_params(axis="x", labelsize=13)

for i in range(len(summary_df)):
    segments = [
        (b[i], 0, pct_b[i]),
        (t[i], b[i], pct_t[i]),
        (c[i], b[i] + t[i], pct_c[i]),
    ]

    for width, left, pct in segments:
        if width > max_total * 0.06:
            ax1.text(
                left + width / 2,
                i,
                f"{human_format(width)}\n({pct:.1f}%)",
                ha="center",
                va="center",
                fontsize=11
            )

    ax1.text(
        totals[i] + max_total * 0.015,
        i,
        f"Mean total: {human_format(totals[i])}\n(n={int(summary_df.loc[i, 'n_simulations'])})",
        va="center",
        fontsize=12
    )

ax2.barh(y, pct_b, color=color_b, edgecolor="black", linewidth=0.8, label="Method B dominant")
ax2.barh(y, pct_t, left=pct_b, color=color_t, edgecolor="black", linewidth=0.8, label="Contested / near parity")
ax2.barh(y, pct_c, left=pct_b + pct_t, color=color_c, edgecolor="black", linewidth=0.8, label="Method C dominant")

ax2.set_yticks(y)
ax2.set_yticklabels(summary_df["scenario"])
ax2.set_xlabel("Mean share of included time across simulations (%)", fontsize=13)
ax2.set_title("Relative composition", fontsize=13)
ax2.set_xlim(0, 100)
ax2.grid(True, axis="x", alpha=0.15)

ax2.tick_params(axis="y", labelsize=13)
ax2.tick_params(axis="x", labelsize=13)

for i in range(len(summary_df)):
    segments_pct = [
        (pct_b[i], 0),
        (pct_t[i], pct_b[i]),
        (pct_c[i], pct_b[i] + pct_t[i]),
    ]

    for width, left in segments_pct:
        if width > 8:
            ax2.text(
                left + width / 2,
                i,
                f"{width:.1f}%",
                ha="center",
                va="center",
                fontsize=11
            )

handles, labels = ax1.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    fontsize=13
)

fig.suptitle("Time spent in regulatory regimes across scenarios\n(mean across simulations)", y=0.98, fontsize=13)
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig('time_spent_chart.png', dpi=900, bbox_inches='tight')
plt.show()

print(summary_df)
print(all_df.head())

