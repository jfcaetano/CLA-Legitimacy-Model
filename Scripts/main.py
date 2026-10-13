#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:33:57 2026

@author: jfcaetano
"""

import numpy as np
import networkx as nx
import pandas as pd
import itertools
import os
from tqdm import tqdm

from belief import Mode
from simulator import Simulator
from datetime import datetime

# -------------------------
# Core simulation settings
# -------------------------
N = 100
T = 3000000
S = 5

mode = Mode.Default
shareTimeLimit = np.inf
delay = 0
singleSource = False
samePartition = None

graph = nx.watts_strogatz_graph
network_name = "SmallWorlds"

# Network parameters
nx_params = {"n": N, "k": 8, "p": 0.01}

experiment = "Watts Strogatz Model_" + datetime.now().strftime("%Y%m%d_%H%M%S")

save = True

# -------------------------
# Stability settings
# -------------------------

STOP_AT_STABILITY = True
STABLE_WINDOW = 1000
STABILITY_ATOL = 1e-9

# Phase 1: early prohibition / overconfident legal reading
#(0.25, 0.10, 0.50, 0.65)

# Phase 2: dispute / recalibration
#(0.40, 0.08, 0.30, 0.55)

# Phase 3: post-clarification / refinement
#(0.70, 0.05, 0.30, 0.45)

# -------------------------
# Scenario grid
# -------------------------

explainability_vals = [0.75]
controversy_growth_vals = [0.10]
trust_sensitivity_vals = [0.30]
fallback_threshold_vals = [0.25]

# Fixed in this run
controversy_decay = 0.05

# Output folders
master_output_dir = os.path.join("./Output", experiment, "grid_runs")
os.makedirs(master_output_dir, exist_ok=True)

results = []


# Run all scenarios

scenario_counter = 0
total_scenarios = (
    len(explainability_vals)
    * len(controversy_growth_vals)
    * len(trust_sensitivity_vals)
    * len(fallback_threshold_vals)
)



for expl, cg, ts, ft in tqdm(
    itertools.product(
        explainability_vals,
        controversy_growth_vals,
        trust_sensitivity_vals,
        fallback_threshold_vals,
    ),
    total=total_scenarios,
    desc="Simulation Progress",
    unit="scenario",
):
    
    
    scenario_counter += 1
    scenario_name = f"expl={expl}_cg={cg}_ts={ts}_ft={ft}"

    print(f"[{scenario_counter}/{total_scenarios}] Running scenario: {scenario_name}")

    sim = Simulator(
        N=N,
        S=S,
        T=T,
        graph=graph,
        nx_params=nx_params,
        sharingMode=mode,
        shareTimeLimit=shareTimeLimit,
        delay=delay,
        singleSource=singleSource,
        samePartition=samePartition,
        method_c_explainability=expl,
        controversy_growth=cg,
        controversy_decay=controversy_decay,
        trust_sensitivity=ts,
        fallback_threshold=ft,
        stop_at_stability=STOP_AT_STABILITY,
        stable_window=STABLE_WINDOW,
        stability_atol=STABILITY_ATOL,
    )

    data = sim.runSimulation(
        save=save,
        experiment=experiment,
        subexperiment=scenario_name,
        network_name=network_name,
        nx_params=nx_params,
        export_csv=True,
    )

    avg_agent, sd_agent, frac_mean, frac_sd, belief_dist, regulation_summary = data
    temporal_data = (avg_agent, sd_agent, frac_mean, frac_sd)


    sim.visBeliefsOverTime(
        data=temporal_data,
        experiment=experiment,
        subexperiment=scenario_name,
        network_name=network_name,
        nx_params=nx_params,
        save=save,
        plot_sd=False,
    )


    stable_neutral = regulation_summary.get("stable_neutral_avg", np.nan)
    stable_method_a = regulation_summary.get("stable_method_a_avg", np.nan)
    stable_method_b = regulation_summary.get("stable_method_b_avg", np.nan)
    stable_method_c = regulation_summary.get("stable_method_c_avg", np.nan)

    stable_neutral_sd = regulation_summary.get("stable_neutral_sd", np.nan)
    stable_method_a_sd = regulation_summary.get("stable_method_a_sd", np.nan)
    stable_method_b_sd = regulation_summary.get("stable_method_b_sd", np.nan)
    stable_method_c_sd = regulation_summary.get("stable_method_c_sd", np.nan)

    avg_stable_time = regulation_summary.get("stable_time_avg", np.nan)
    sd_stable_time = regulation_summary.get("stable_time_sd", np.nan)
    avg_steps_run = regulation_summary.get("steps_run_avg", np.nan)
    sd_steps_run = regulation_summary.get("steps_run_sd", np.nan)

    print(
        f"    Avg stable time: {avg_stable_time:.2f} | "
        f"Avg steps run: {avg_steps_run:.2f} | "
        f"Stable Method C: {stable_method_c:.4f}"
    )


    results.append({
        "scenario": scenario_name,
        "method_c_explainability": expl,
        "controversy_growth": cg,
        "controversy_decay": controversy_decay,
        "trust_sensitivity": ts,
        "fallback_threshold": ft,
        "stable_window": STABLE_WINDOW,
        "stable_time_avg": avg_stable_time,
        "stable_time_sd": sd_stable_time,
        "steps_run_avg": avg_steps_run,
        "steps_run_sd": sd_steps_run,
        "stable_neutral": stable_neutral,
        "stable_method_a": stable_method_a,
        "stable_method_b": stable_method_b,
        "stable_method_c": stable_method_c,
        "stable_neutral_sd": stable_neutral_sd,
        "stable_method_a_sd": stable_method_a_sd,
        "stable_method_b_sd": stable_method_b_sd,
        "stable_method_c_sd": stable_method_c_sd,
        "avg_neutral": avg_agent[0],
        "avg_method_a": avg_agent[1],
        "avg_method_b": avg_agent[2],
        "avg_method_c": avg_agent[3],
        "trust_avg": regulation_summary["trust_avg"],
        "trust_sd": regulation_summary["trust_sd"],
        "controversy_avg": regulation_summary["controversy_avg"],
        "controversy_sd": regulation_summary["controversy_sd"],
        "fallback_strength_at_stable": stable_method_b - stable_method_c,
        "legitimacy_score": regulation_summary["trust_avg"] - regulation_summary["controversy_avg"],
    })


# Save full summary

summary_df = pd.DataFrame(results)

summary_csv = os.path.join(master_output_dir, "scenario_summary_stable_state.csv")
summary_df.to_csv(summary_csv, index=False)

# Best and worst cases

best_method_c = summary_df.sort_values(
    by=["stable_method_c", "trust_avg", "controversy_avg"],
    ascending=[False, False, True]
).head(10)

worst_method_c = summary_df.sort_values(
    by=["stable_method_c", "trust_avg", "controversy_avg"],
    ascending=[True, True, False]
).head(10)

best_fallback_b = summary_df.sort_values(
    by=["stable_method_b", "fallback_strength_at_stable", "trust_avg"],
    ascending=[False, False, True]
).head(10)

worst_fallback_b = summary_df.sort_values(
    by=["stable_method_b", "fallback_strength_at_stable", "trust_avg"],
    ascending=[True, True, False]
).head(10)

best_trust = summary_df.sort_values(
    by=["trust_avg", "controversy_avg", "stable_method_c"],
    ascending=[False, True, False]
).head(10)

worst_trust = summary_df.sort_values(
    by=["trust_avg", "controversy_avg", "stable_method_c"],
    ascending=[True, False, True]
).head(10)

best_controversy = summary_df.sort_values(
    by=["controversy_avg", "trust_avg", "stable_method_b"],
    ascending=[False, True, False]
).head(10)

lowest_controversy = summary_df.sort_values(
    by=["controversy_avg", "trust_avg", "stable_method_c"],
    ascending=[True, False, False]
).head(10)

best_legitimacy = summary_df.sort_values(
    by=["legitimacy_score", "stable_method_c", "trust_avg"],
    ascending=[False, False, False]
).head(10)

worst_legitimacy = summary_df.sort_values(
    by=["legitimacy_score", "stable_method_c", "trust_avg"],
    ascending=[True, True, True]
).head(10)


best_method_c.to_csv(os.path.join(master_output_dir, "top10_method_c_stable.csv"), index=False)
worst_method_c.to_csv(os.path.join(master_output_dir, "bottom10_method_c_stable.csv"), index=False)

best_fallback_b.to_csv(os.path.join(master_output_dir, "top10_fallback_b_stable.csv"), index=False)
worst_fallback_b.to_csv(os.path.join(master_output_dir, "bottom10_fallback_b_stable.csv"), index=False)

best_trust.to_csv(os.path.join(master_output_dir, "top10_trust_stable.csv"), index=False)
worst_trust.to_csv(os.path.join(master_output_dir, "bottom10_trust_stable.csv"), index=False)

best_controversy.to_csv(os.path.join(master_output_dir, "top10_controversy_stable.csv"), index=False)
lowest_controversy.to_csv(os.path.join(master_output_dir, "bottom10_controversy_stable.csv"), index=False)

best_legitimacy.to_csv(os.path.join(master_output_dir, "top10_legitimacy_stable.csv"), index=False)
worst_legitimacy.to_csv(os.path.join(master_output_dir, "bottom10_legitimacy_stable.csv"), index=False)


print("\nEnd")
