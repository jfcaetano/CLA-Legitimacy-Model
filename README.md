# CLA Legitimacy Model

**Agent-based simulation of belief diffusion, trust, controversy, and regulatory legitimacy on social networks**

This repository contains Python code, simulation results, and videos for an agent-based model that explores how competing methods (*A*, *B*, and *C*) spread through a network and how adoption of Method C interacts with an aggregate measure of **trust**, **controversy**, and a probabilistic **fallback to Method B**.

The model combines a Watts–Strogatz small-world network, pairwise agent interactions, time-dependent belief transitions, and an aggregate regulatory-state feedback mechanism. It supports repeated stochastic runs, configurable parameter sweeps, optional early stopping, and exports of belief trajectories and simulation summaries.

> **Scope:** “Trust,” “controversy,” and the reported “legitimacy score” are quantities defined by the simulation. They are modeling proxies, not direct empirical measurements or validated indicators of legal or institutional legitimacy. Interpretations of Methods A, B, and C should be supplied by the research context; their labels alone do not define specific real-world policies.

## Repository structure

```text
CLA-Legitimacy-Model/
├── Scripts/
│   ├── main.py                      # Experiment settings and scenario execution
│   ├── simulator.py                 # Repeated simulations, stability checks, CSV/plot output
│   ├── model.py                     # KnowledgeModel: agent setup, trust, controversy
│   ├── agent.py                     # PopAgent: belief transitions, adoption, fallback
│   ├── belief.py                    # Belief and Mode enumerations
│   ├── simpleScheduler.py           # Pair-selection and timestep scheduling
│   ├── simpleLogger.py              # Belief/interaction histories
│   └── Highest_PR_Method_80-3P.csv  # Input used to derive transition thresholds
├── Results/                         # Research results provided in the repository
└── Simulation Videos/               # Simulation videos provided in the repository
```

The paths above identify the core files examined for this documentation. For the complete current contents, see the [repository on GitHub](https://github.com/jfcaetano/CLA-Legitimacy-Model).

## Installation

Clone the repository and install the required libraries:

```bash
git clone https://github.com/jfcaetano/CLA-Legitimacy-Model.git
cd CLA-Legitimacy-Model
python -m pip install numpy pandas networkx matplotlib tqdm "mesa>=2,<3"
```

**Mesa compatibility:** This implementation imports `BaseScheduler` from `mesa.time` and constructs agents with the older `Agent(unique_id, model)` interface. These are legacy Mesa APIs; Mesa 3 introduced incompatible changes. A Mesa **2.x** environment is therefore the appropriate starting point for reproducing this version of the code. Specific package versions were not pinned in the inspected scripts.

You can run the code in an ordinary Python environment or in **Spyder**. Make sure the working directory contains `Highest_PR_Method_80-3P.csv` when running: `agent.py` loads this file by a **relative filename at import time**.

## Running an experiment

From the repository root:

```bash
cd Scripts
python main.py
```

In Spyder, open `Scripts/main.py`, set Spyder's **working directory to `Scripts/`**, then run the script.

The current `main.py` defines these defaults:

| Parameter | Current value | Description |
| --- | ---: | --- |
| `N` | `100` | Number of agents (network nodes) |
| `T` | `3_000_000` | Maximum recorded time points per realization, including the initial state |
| `S` | `5` | Independent stochastic realizations per scenario |
| Network | `nx.watts_strogatz_graph` | Small-world network generator |
| `k` | `8` | Initial nearest-neighbor degree |
| `p` | `0.01` | Edge rewiring probability |
| `shareTimeLimit` | `np.inf` | No finite sharing-time limit in the current configuration |
| `delay` | `0` | No delayed introduction of Method B |
| `singleSource` | `False` | Method introductions are not restricted to one source |
| `samePartition` | `None` | No partition-based source constraint |
| `STOP_AT_STABILITY` | `True` | Enable plateau-based early stopping |
| `STABLE_WINDOW` | `1000` | Number of recorded time points evaluated for a plateau |
| `STABILITY_ATOL` | `1e-9` | Maximum allowed range for each belief fraction over the plateau window |
| `post_stable_steps` | `1000` | Additional steps after a plateau is confirmed (default in `Simulator`) |

The current script sets one value for each scenario parameter:

| Parameter | Current value | Meaning in the code |
| --- | ---: | --- |
| `method_c_explainability` | `0.75` | Reduces controversy pressure associated with Method C |
| `controversy_growth` | `0.10` | Scales controversy increases |
| `controversy_decay` | `0.05` | Subtracted from controversy each step |
| `trust_sensitivity` | `0.30` | Scales the decrease in trust with controversy |
| `fallback_threshold` | `0.25` | Trust threshold for possible C-to-B fallback |

**Scenario count:** Even if the code/comments refer to an 81-scenario grid, the current single-element parameter lists produce **1 scenario × 5 realizations**. To perform a grid search, add more values to the four parameter lists in `main.py`.

## Model architecture

### 1. Network and initial beliefs

Each realization generates a new undirected Watts–Strogatz small-world graph with `N` nodes. Agents occupy nodes and can interact with neighboring agents.

Four belief states are defined in `belief.py`:

| Code | State |
| ---: | --- |
| `0` | Neutral |
| `1` | Method A |
| `2` | Method B |
| `3` | Method C |

Agents begin in the Neutral state, except for deliberately introduced seed beliefs. Agent `0` is initialized with Method A. Depending on `singleSource`, `samePartition`, and `delay`, methods B and C are assigned to additional agents during initialization (or B can be introduced later). In the current configuration, B and C are seeded at initialization, with their recipients selected using Python's random generator. A newly assigned belief can replace an earlier seed if the same agent is chosen.

### 2. Scheduling and interactions

One call to `KnowledgeModel.step()` corresponds to one model timestep. The scheduler:

1. Increments the clock and current-belief duration for **every** agent.
2. Selects one agent uniformly at random and then selects one of that agent's network neighbors at random.
3. Calls the update function for **both** agents in the selected pair.
4. Logs agent states/interactions and advances the scheduler's time counter.
5. Updates the model-level controversy and trust after the pair interaction.

Thus a timestep is **one interacting pair**, not a sweep in which every agent interacts. The selected pair is stochastic, and changes may occur in both directions within that pair.

### 3. Transition thresholds from the CSV input

`agent.py` loads `Scripts/Highest_PR_Method_80-3P.csv` and extracts the **first** `Time` value for each `Choice` (`A`, `B`, and `C`):

- `NEUTRAL_TO_METHOD_A_THRESHOLD` — first row with `Choice == "A"`.
- `METHOD_A_TO_METHOD_B_THRESHOLD` — first row with `Choice == "B"`.
- `METHOD_B_TO_METHOD_C_THRESHOLD` — first row with `Choice == "C"`.

For the version inspected, these values are **0**, **100**, and **650**, respectively. The model uses them as thresholds against an agent's `beliefTime`, i.e., time elapsed **since that agent last changed belief**. The CSV's other columns (such as `PR_A`, `PR_B`, and `PR_C`) are not directly used by the agent transition code; their role in constructing the input CSV is upstream of this implementation.

Belief changes are interaction-dependent. In broad terms, agents can acquire A or B when interacting with a holder of that method, make time-conditional transitions among A/B/C, and adopt C subject to the trust-and-controversy mechanism described below. See [`agent.py`](Scripts/agent.py) for the exact ordering of transition rules and conditions.

### 4. Controversy and trust

At each timestep, `KnowledgeModel.update_regulation_state()` computes population fractions `f_A`, `f_B`, and `f_C` (with the Neutral population implicitly affecting the disagreement term):

```text
disagreement = 1 - max(f_A, f_B, f_C)
c_pressure  = f_C * (1 - method_c_explainability)

controversy_next = clip(
    controversy + controversy_growth * (c_pressure + disagreement)
    - controversy_decay,
    0, 1
)

trust_next = clip(1 - trust_sensitivity * controversy_next, 0, 1)
```

Here, `clip(x, 0, 1)` bounds `x` to the interval `[0, 1]`. The initial conditions are `controversy = 0` and `trust = 1`. Under this implementation, `trust` is a derived global state based on current controversy; it is not separately estimated from agents.

### 5. Trust-sensitive Method C adoption and fallback

When a transition rule invokes Method C adoption, its probability is:

```text
P(adopt C) = clip(trust * (1 - 0.5 * controversy), 0, 1)
```

If trust falls below `fallback_threshold`, an agent holding Method C can revert to Method B with probability:

```text
P(fallback C -> B) = clip(
    (fallback_threshold - trust) + 0.3 * controversy,
    0, 1
)
```

For `trust >= fallback_threshold`, this fallback check returns `False`. These probabilities are evaluated using Python's `random` module in applicable transition branches; they do not replace the other transition and interaction rules.

## Stability and stopping

The simulator records the four population fractions at `t = 0` and after each subsequent model step. When `STOP_AT_STABILITY=True`, it checks the most recent `STABLE_WINDOW` recorded states and confirms a plateau if **all four fractions** satisfy:

```text
max(fraction over window) - min(fraction over window) <= STABILITY_ATOL
```

After the first confirmation, the realization continues for `post_stable_steps` additional steps, unless the maximum timestep limit is reached first.

**Important distinctions:**

- Plateau stability concerns **population proportions**, not the absence of all agent-level transitions or guarantee of an absorbing state.
- `Stable_Time_Confirmed` refers to the first detected plateau, not necessarily the final simulated timestep.
- `Stable_*` output columns contain the fractions **at the confirmation timestep**, while `Final_Trust` and `Final_Controversy` refer to the **end of the run**.
- If early stopping is disabled or no plateau is detected, the code assigns `stable_time` to the last recorded time point. In those cases, “stable” values should be interpreted as **last-observation values**, not confirmed convergence.

## Outputs

`main.py` creates a timestamped experiment folder, avoiding name collisions between runs executed at different timestamps. Because `simulator.py` uses relative output paths, files are written beneath `Scripts/Output/` when run from `Scripts/`.

```text
Scripts/
└── Output/
    └── Watts Strogatz Model_YYYYMMDD_HHMMSS/
        ├── grid_runs/
        │   └── scenario_summary_stable_state.csv
        └── expl=0.75_cg=0.1_ts=0.3_ft=0.25/
            ├── ..._ratios_until_stop.csv
            ├── ..._stable_summary.csv
            └── plots/
                └── ....png
```

The omitted parts of file names contain simulation and parameter settings.

### Timestep-level ratios CSV

`*_ratios_until_stop.csv` contains one row for every recorded timestep in each realization:

| Column | Meaning |
| --- | --- |
| `Simulation` | Realization identifier (1 through `S`) |
| `Timestep` | Recorded time point, beginning at 0 |
| `Neutral` | Fraction of agents with no adopted method |
| `Method_A` | Fraction holding Method A |
| `Method_B` | Fraction holding Method B |
| `Method_C` | Fraction holding Method C |

The four fractions sum to approximately 1 (subject to floating-point precision).

### Per-realization summary CSV

`*_stable_summary.csv` includes the realization number, stability confirmation time, steps run, stopping indicator, post-stability setting, five regulation parameters, the four belief fractions at the selected stable point, and final trust and controversy. Its exact column names are defined in `simulator.py`.

### Scenario-level summary CSV

`grid_runs/scenario_summary_stable_state.csv` contains parameter values and statistics aggregated across the `S` realizations, including means and standard deviations of stable times, steps run, stable belief fractions, trust, and controversy.

The main script also defines:

```text
fallback_strength_at_stable = stable_method_b - stable_method_c
legitimacy_score            = trust_avg - controversy_avg
```

The legitimacy score is a simple constructed index; it should not be interpreted as an externally calibrated legitimacy measurement. In this version of `main.py`, the exported scenario ranking tables shown in some alternative scripts are **not** generated; the script writes the scenario summary.

### Plots

`Simulator.visBeliefsOverTime()` produces plots of the mean population belief fractions over time (Neutral, A, B, C), with optional standard-deviation lines and configurable smoothing. With `save=True`, plots are saved as PNG files. `matplotlib.pyplot.show()` also displays the plot in the active Python environment.

## Extracting the final values of each realization

Run the following in Spyder or another Python environment, replacing the path with an exported ratios CSV:

```python
import pandas as pd

csv_file = r"path/to/your_ratios_until_stop.csv"
df = pd.read_csv(csv_file)

final = (
    df.loc[df.groupby("Simulation")["Timestep"].idxmax(),
           ["Simulation", "Timestep", "Neutral", "Method_A", "Method_B", "Method_C"]]
    .sort_values("Simulation")
)

print(final.to_string(index=False))
print("\nMean final population fractions:")
print(final[["Neutral", "Method_A", "Method_B", "Method_C"]].mean())
```

This extracts the *last recorded* point from each realization, which can differ from the plateau confirmation point in the stability summary.

## Reproducibility and computational considerations

- **Stochasticity:** Network rewiring, initial method seeding, pair selection, and selected adoption/fallback transitions involve randomness. The inspected `main.py` does **not** set fixed random seeds. Repeated executions can therefore yield different results even with identical parameters.
- **Output preservation:** Experiment folder names include the current timestamp to the second. Runs launched in different seconds normally produce separate folders; repeated runs with an identical timestamp string could still target the same paths.
- **Runtime:** A default scenario requests up to 5 × (3,000,000 − 1) interacting-pair timesteps. Early stopping may reduce this amount substantially.
- **Memory:** The scheduler's logger retains per-agent belief histories and interaction records during a run. This can become expensive at millions of timesteps, **even when** `compute_agent_history_stats=False` (which disables calculation of certain optional statistics but does not turn off `SimpleLogger.log`). If a long simulation exhausts memory, this logging design is a key place to investigate.
- **Progress reporting:** The code uses `tqdm` to report time-step progress and outer scenario progress.
- **Environment:** Record the Python and dependency versions, random seeds (if introduced), network parameters, model parameters, input CSV version, and the commit hash for publication-grade replication.

## Where to read the implementation

- [`Scripts/main.py`](Scripts/main.py): scenario configuration, execution, and summary export.
- [`Scripts/model.py`](Scripts/model.py): agent initialization, regulatory state update, and one model step.
- [`Scripts/agent.py`](Scripts/agent.py): transition conditions, trust-sensitive Method C adoption, and fallback.
- [`Scripts/belief.py`](Scripts/belief.py): belief state identifiers.
- [`Scripts/simpleScheduler.py`](Scripts/simpleScheduler.py): random neighboring-pair activation.
- [`Scripts/simpleLogger.py`](Scripts/simpleLogger.py): recorded belief and interaction histories.
- [`Scripts/simulator.py`](Scripts/simulator.py): repeated trials, early stopping, statistics, CSV export, and plotting.
- [`Scripts/Highest_PR_Method_80-3P.csv`](Scripts/Highest_PR_Method_80-3P.csv): threshold input derived from the `Choice` and `Time` columns.

## Project status and citation

This is a research simulation repository. Its outputs describe the behavior of the **specified model assumptions**, not causal evidence about real-world regulatory systems without additional calibration and validation.

For attribution, link to the repository: [jfcaetano/CLA-Legitimacy-Model](https://github.com/jfcaetano/CLA-Legitimacy-Model). If this model accompanies a publication, add the publication's bibliographic reference and DOI here.
