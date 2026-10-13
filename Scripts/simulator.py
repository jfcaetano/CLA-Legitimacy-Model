import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import KnowledgeModel
from belief import Belief

plt.rcParams["font.family"] = "Arial"


class Simulator:
    """Run repeated simulations of the knowledge model, with optional early stopping."""

    def __init__(
        self,
        N,
        S,
        T,
        graph,
        nx_params,
        sharingMode,
        shareTimeLimit,
        delay,
        singleSource,
        samePartition,
        method_c_explainability=0.35,
        controversy_growth=0.08,
        controversy_decay=0.03,
        trust_sensitivity=0.5,
        fallback_threshold=0.55,
        stop_at_stability=True,
        stable_window=100,
        stability_atol=0.01,          
        post_stable_steps=1000,
        plot_smoothing_window=1001,   
        compute_agent_history_stats=False,
    ):
        self.N = N
        self.S = S
        self.T = T
        self.graph = graph
        self.nx_params = nx_params
        self.mode = sharingMode
        self.shareTimeLimit = shareTimeLimit
        self.delay = delay
        self.singleSource = singleSource
        self.samePartition = samePartition

        self.method_c_explainability = method_c_explainability
        self.controversy_growth = controversy_growth
        self.controversy_decay = controversy_decay
        self.trust_sensitivity = trust_sensitivity
        self.fallback_threshold = fallback_threshold

        self.stop_at_stability = stop_at_stability
        self.stable_window = stable_window
        self.stability_atol = stability_atol
        self.post_stable_steps = post_stable_steps
        self.plot_smoothing_window = plot_smoothing_window
        self.compute_agent_history_stats = compute_agent_history_stats

    def _get_population_ratios(self, model):
        """Return current population ratios in the order Neutral, A, B, C."""
        beliefs = [agent.belief for agent in model.schedule.agents]
        n = len(beliefs)

        neutral_ratio = sum(1 for b in beliefs if b == Belief.Neutral) / n
        method_a_ratio = sum(1 for b in beliefs if b == Belief.Method_A) / n
        method_b_ratio = sum(1 for b in beliefs if b == Belief.Method_B) / n
        method_c_ratio = sum(1 for b in beliefs if b == Belief.Method_C) / n

        return np.array(
            [neutral_ratio, method_a_ratio, method_b_ratio, method_c_ratio],
            dtype=float,
        )

    def _compute_agent_history_stats_from_logs(self, logs):
        """Optional and slower: compute average per-agent time spent in each belief state."""
        df_belief = pd.DataFrame.from_dict(logs[0])
        values = df_belief.values

        neutral_avg = np.mean(np.sum(values == 0, axis=0))
        method_a_avg = np.mean(np.sum(values == 1, axis=0))
        method_b_avg = np.mean(np.sum(values == 2, axis=0))
        method_c_avg = np.mean(np.sum(values == 3, axis=0))

        return neutral_avg, method_a_avg, method_b_avg, method_c_avg

    def runModel(self, network):
        """
        Run a single model realization.

        Stability is defined as a plateau:
        over the last `stable_window` steps, each ratio stays within
        a band no wider than `stability_atol`.

        After stability is confirmed, the model continues for
        `post_stable_steps` more steps.
        """
        model = KnowledgeModel(
            network=network,
            sharingMode=self.mode,
            sharetime=self.shareTimeLimit,
            delay=self.delay,
            singleSource=self.singleSource,
            samePartition=self.samePartition,
            method_c_explainability=self.method_c_explainability,
            controversy_growth=self.controversy_growth,
            controversy_decay=self.controversy_decay,
            trust_sensitivity=self.trust_sensitivity,
            fallback_threshold=self.fallback_threshold,
        )

        ratio_history = []
        stable_time = None

        # Record initial state at t = 0
        current_ratio = self._get_population_ratios(model)
        ratio_history.append(current_ratio.copy())

        
        for t in tqdm(
            range(1, self.T),
            desc="Simulation Time Steps",
            unit="step",
            leave=True,
            ):
            model.step()
            current_ratio = self._get_population_ratios(model)
            ratio_history.append(current_ratio.copy())

            # Plateau-based stability detection
            if (
                self.stop_at_stability
                and stable_time is None
                and len(ratio_history) >= self.stable_window
            ):
                recent = np.asarray(ratio_history[-self.stable_window:], dtype=float)
                band = np.max(recent, axis=0) - np.min(recent, axis=0)

                if np.all(band <= self.stability_atol):
                    stable_time = t  

            # Continue for post_stable_steps after confirmation
            if (
                self.stop_at_stability
                and stable_time is not None
                and t >= stable_time + self.post_stable_steps
            ):
                break

        if stable_time is None:
            stable_time = len(ratio_history) - 1

        logs = model.logs() if self.compute_agent_history_stats else None

        return {
            "logs": logs,
            "model": model,
            "ratio_history": np.asarray(ratio_history, dtype=float),
            "stable_time": int(stable_time),
            "steps_run": len(ratio_history),
            "reached_stability": stable_time < (len(ratio_history) - 1),
        }

    def runSimulation(
        self,
        save,
        experiment,
        subexperiment,
        network_name,
        nx_params,
        export_csv=True,
    ):
        directory = "./Output/" + experiment + "/" + subexperiment
        name = network_name + "_" + "_".join([f"{k}={v}" for k, v in nx_params.items()])

        if save or export_csv:
            os.makedirs(directory, exist_ok=True)

        ratio_logs = []
        stable_state_logs = []

        # Keep these only if stats are enabled
        num_neutral_per_agent = np.full(self.S, np.nan)
        num_Method_A_per_agent = np.full(self.S, np.nan)
        num_Method_B_per_agent = np.full(self.S, np.nan)
        num_Method_C_per_agent = np.full(self.S, np.nan)

        trust_final = np.full(self.S, np.nan)
        controversy_final = np.full(self.S, np.nan)
        stable_times = np.full(self.S, np.nan)
        steps_run_all = np.full(self.S, np.nan)

        stable_neutral_runs = np.full(self.S, np.nan)
        stable_method_a_runs = np.full(self.S, np.nan)
        stable_method_b_runs = np.full(self.S, np.nan)
        stable_method_c_runs = np.full(self.S, np.nan)

        all_ratio_histories = []

        for s in range(self.S):
            network = self.graph(**self.nx_params)
            run_output = self.runModel(network=network)

            logs = run_output["logs"]
            model = run_output["model"]
            ratio_history = run_output["ratio_history"]
            stable_time = int(run_output["stable_time"])
            steps_run = int(run_output["steps_run"])
            reached_stability = bool(run_output["reached_stability"])

            all_ratio_histories.append(ratio_history)
            stable_times[s] = stable_time
            steps_run_all[s] = steps_run

            stable_neutral_runs[s] = ratio_history[stable_time, 0]
            stable_method_a_runs[s] = ratio_history[stable_time, 1]
            stable_method_b_runs[s] = ratio_history[stable_time, 2]
            stable_method_c_runs[s] = ratio_history[stable_time, 3]

            sim_ids = np.full(steps_run, s + 1)
            timesteps = np.arange(steps_run)
            run_rows = np.column_stack([sim_ids, timesteps, ratio_history])
            ratio_logs.extend(run_rows.tolist())

            stable_state_logs.append(
                [
                    s + 1,
                    stable_time,
                    steps_run,
                    reached_stability,
                    self.post_stable_steps,
                    self.method_c_explainability,
                    self.controversy_growth,
                    self.controversy_decay,
                    self.trust_sensitivity,
                    self.fallback_threshold,
                    ratio_history[stable_time, 0],
                    ratio_history[stable_time, 1],
                    ratio_history[stable_time, 2],
                    ratio_history[stable_time, 3],
                    getattr(model, "trust", np.nan),
                    getattr(model, "controversy", np.nan),
                ]
            )

            if self.compute_agent_history_stats and logs is not None:
                (
                    num_neutral_per_agent[s],
                    num_Method_A_per_agent[s],
                    num_Method_B_per_agent[s],
                    num_Method_C_per_agent[s],
                ) = self._compute_agent_history_stats_from_logs(logs)

            trust_final[s] = getattr(model, "trust", np.nan)
            controversy_final[s] = getattr(model, "controversy", np.nan)

        max_len = max(hist.shape[0] for hist in all_ratio_histories)

        neutral_per_timestep = np.full((self.S, max_len), np.nan)
        Method_A_per_timestep = np.full((self.S, max_len), np.nan)
        Method_B_per_timestep = np.full((self.S, max_len), np.nan)
        Method_C_per_timestep = np.full((self.S, max_len), np.nan)

        for s, hist in enumerate(all_ratio_histories):
            run_len = hist.shape[0]
            neutral_per_timestep[s, :run_len] = hist[:, 0]
            Method_A_per_timestep[s, :run_len] = hist[:, 1]
            Method_B_per_timestep[s, :run_len] = hist[:, 2]
            Method_C_per_timestep[s, :run_len] = hist[:, 3]

        if export_csv:
            ratio_df = pd.DataFrame(
                ratio_logs,
                columns=[
                    "Simulation",
                    "Timestep",
                    "Neutral",
                    "Method_A",
                    "Method_B",
                    "Method_C",
                ],
            )

            stable_df = pd.DataFrame(
                stable_state_logs,
                columns=[
                    "Simulation",
                    "Stable_Time_Confirmed",
                    "Steps_Run",
                    "Reached_Stability",
                    "Post_Stable_Steps",
                    "method_c_explainability",
                    "controversy_growth",
                    "controversy_decay",
                    "trust_sensitivity",
                    "fallback_threshold",
                    "Stable_Neutral",
                    "Stable_Method_A",
                    "Stable_Method_B",
                    "Stable_Method_C",
                    "Final_Trust",
                    "Final_Controversy",
                ],
            )

            filename_base = (
                "/N={N}-T={T}-S={S}-shr={shr}-dly={dly}-post={post}-"
                "expl={expl}-cg={cg}-cd={cd}-ts={ts}-ft={ft}-{name}"
            ).format(
                N=self.N,
                T=self.T,
                S=self.S,
                shr=self.shareTimeLimit,
                dly=self.delay,
                post=self.post_stable_steps,
                expl=self.method_c_explainability,
                cg=self.controversy_growth,
                cd=self.controversy_decay,
                ts=self.trust_sensitivity,
                ft=self.fallback_threshold,
                name=name,
            )

            ratio_df.to_csv(directory + filename_base + "_ratios_until_stop.csv", index=False)
            stable_df.to_csv(directory + filename_base + "_stable_summary.csv", index=False)

        # Avoid warnings when stats are disabled
        if self.compute_agent_history_stats:
            neutral_per_agent_avg = np.nanmean(num_neutral_per_agent)
            neutral_per_agent_sd = np.nanstd(num_neutral_per_agent)

            Method_A_per_agent_avg = np.nanmean(num_Method_A_per_agent)
            Method_A_per_agent_sd = np.nanstd(num_Method_A_per_agent)

            Method_B_per_agent_avg = np.nanmean(num_Method_B_per_agent)
            Method_B_per_agent_sd = np.nanstd(num_Method_B_per_agent)

            Method_C_per_agent_avg = np.nanmean(num_Method_C_per_agent)
            Method_C_per_agent_sd = np.nanstd(num_Method_C_per_agent)
        else:
            neutral_per_agent_avg = np.nan
            neutral_per_agent_sd = np.nan
            Method_A_per_agent_avg = np.nan
            Method_A_per_agent_sd = np.nan
            Method_B_per_agent_avg = np.nan
            Method_B_per_agent_sd = np.nan
            Method_C_per_agent_avg = np.nan
            Method_C_per_agent_sd = np.nan

        frac_neutral_per_timestep = np.nanmean(neutral_per_timestep, axis=0)
        frac_neutral_per_timestep_sd = np.nanstd(neutral_per_timestep, axis=0)

        frac_Method_A_per_timestep = np.nanmean(Method_A_per_timestep, axis=0)
        frac_Method_A_per_timestep_sd = np.nanstd(Method_A_per_timestep, axis=0)

        frac_Method_B_per_timestep = np.nanmean(Method_B_per_timestep, axis=0)
        frac_Method_B_per_timestep_sd = np.nanstd(Method_B_per_timestep, axis=0)

        frac_Method_C_per_timestep = np.nanmean(Method_C_per_timestep, axis=0)
        frac_Method_C_per_timestep_sd = np.nanstd(Method_C_per_timestep, axis=0)

        neutral_dist = np.array([hist[int(st), 0] for hist, st in zip(all_ratio_histories, stable_times)])
        Method_A_dist = np.array([hist[int(st), 1] for hist, st in zip(all_ratio_histories, stable_times)])
        Method_B_dist = np.array([hist[int(st), 2] for hist, st in zip(all_ratio_histories, stable_times)])
        Method_C_dist = np.array([hist[int(st), 3] for hist, st in zip(all_ratio_histories, stable_times)])

        regulation_summary = {
            "trust_avg": np.nanmean(trust_final),
            "trust_sd": np.nanstd(trust_final),
            "controversy_avg": np.nanmean(controversy_final),
            "controversy_sd": np.nanstd(controversy_final),
            "stable_time_avg": np.nanmean(stable_times),
            "stable_time_sd": np.nanstd(stable_times),
            "steps_run_avg": np.nanmean(steps_run_all),
            "steps_run_sd": np.nanstd(steps_run_all),
            "stable_window": self.stable_window,
            "post_stable_steps": self.post_stable_steps,
            "stable_neutral_avg": np.nanmean(stable_neutral_runs),
            "stable_neutral_sd": np.nanstd(stable_neutral_runs),
            "stable_method_a_avg": np.nanmean(stable_method_a_runs),
            "stable_method_a_sd": np.nanstd(stable_method_a_runs),
            "stable_method_b_avg": np.nanmean(stable_method_b_runs),
            "stable_method_b_sd": np.nanstd(stable_method_b_runs),
            "stable_method_c_avg": np.nanmean(stable_method_c_runs),
            "stable_method_c_sd": np.nanstd(stable_method_c_runs),
        }

        avg_per_agent = (
            neutral_per_agent_avg,
            Method_A_per_agent_avg,
            Method_B_per_agent_avg,
            Method_C_per_agent_avg,
        )

        sd_per_agent = (
            neutral_per_agent_sd,
            Method_A_per_agent_sd,
            Method_B_per_agent_sd,
            Method_C_per_agent_sd,
        )

        frac_belief_mean = (
            frac_neutral_per_timestep,
            frac_Method_A_per_timestep,
            frac_Method_B_per_timestep,
            frac_Method_C_per_timestep,
        )

        frac_belief_sd = (
            frac_neutral_per_timestep_sd,
            frac_Method_A_per_timestep_sd,
            frac_Method_B_per_timestep_sd,
            frac_Method_C_per_timestep_sd,
        )

        belief_dist = (
            neutral_dist,
            Method_A_dist,
            Method_B_dist,
            Method_C_dist,
        )

        return (
            avg_per_agent,
            sd_per_agent,
            frac_belief_mean,
            frac_belief_sd,
            belief_dist,
            regulation_summary,
        )

    def visBeliefsOverTime(
        self,
        data,
        experiment,
        subexperiment,
        network_name,
        nx_params,
        save,
        plot_sd=False,
    ):
        _, _, frac_belief_mean, frac_belief_sd = data

        neutral_mean, Method_A_mean, Method_B_mean, Method_C_mean = frac_belief_mean
        neutral_sd, Method_A_sd, Method_B_sd, Method_C_sd = frac_belief_sd

        series_len = len(neutral_mean)
        x = np.arange(series_len)

        fig, ax = plt.subplots(figsize=(10, 6))

        def fill_nans(y):
            y = np.asarray(y, dtype=float)
            if not np.isnan(y).any():
                return y
            valid = ~np.isnan(y)
            if valid.sum() == 0:
                return np.zeros_like(y, dtype=float)
            if valid.sum() == 1:
                return np.full_like(y, y[valid][0], dtype=float)
            return np.interp(np.arange(len(y)), np.where(valid)[0], y[valid])

        def smooth_series(y, window):
            """
            Same-length edge-padded smoothing with no end drop.
            """
            y = fill_nans(y)

            if window is None or window <= 1 or len(y) < window:
                return y.copy()

            # Force odd window so length handling is stable
            if window % 2 == 0:
                window += 1

            kernel = np.ones(window, dtype=float) / window
            pad = window // 2

            # same-length smoothing using edge padding
            y_pad = np.pad(y, (pad, pad), mode="edge")
            y1 = np.convolve(y_pad, kernel, mode="valid")

            y1_pad = np.pad(y1, (pad, pad), mode="edge")
            y2 = np.convolve(y1_pad, kernel, mode="valid")

            # preserve exact endpoints
            y2[0] = y[0]
            y2[-1] = y[-1]

            return y2

        window = self.plot_smoothing_window

        neutral_plot = smooth_series(neutral_mean, window)
        Method_A_plot = smooth_series(Method_A_mean, window)
        Method_B_plot = smooth_series(Method_B_mean, window)
        Method_C_plot = smooth_series(Method_C_mean, window)

        line1, = ax.plot(x, neutral_plot, label="Neutral", color="grey", ls="-", linewidth=2.5)
        line2, = ax.plot(x, Method_A_plot, label="Method A", color="blue", ls="-", linewidth=2.5)
        line3, = ax.plot(x, Method_B_plot, label="Method B", color="orange", ls="-", linewidth=2.5)
        line4, = ax.plot(x, Method_C_plot, label="Method C", color="green", ls="-", linewidth=2.5)

        if plot_sd:
            alpha = 0.15
            ax.plot(x, fill_nans(neutral_mean + neutral_sd), color="grey", ls="--", alpha=alpha, linewidth=1.0)
            ax.plot(x, fill_nans(neutral_mean - neutral_sd), color="grey", ls="--", alpha=alpha, linewidth=1.0)

            ax.plot(x, fill_nans(Method_A_mean + Method_A_sd), color="blue", ls="--", alpha=alpha, linewidth=1.0)
            ax.plot(x, fill_nans(Method_A_mean - Method_A_sd), color="blue", ls="--", alpha=alpha, linewidth=1.0)

            ax.plot(x, fill_nans(Method_B_mean + Method_B_sd), color="orange", ls="--", alpha=alpha, linewidth=1.0)
            ax.plot(x, fill_nans(Method_B_mean - Method_B_sd), color="orange", ls="--", alpha=alpha, linewidth=1.0)

            ax.plot(x, fill_nans(Method_C_mean + Method_C_sd), color="green", ls="--", alpha=alpha, linewidth=1.0)
            ax.plot(x, fill_nans(Method_C_mean - Method_C_sd), color="green", ls="--", alpha=alpha, linewidth=1.0)

        ax.set_xlim(0, max(series_len - 1, 1))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Time", fontsize=14, fontname="Arial")
        ax.set_ylabel("Ratio of population belief", fontsize=14, fontname="Arial")

        ax.legend(
            handles=[line1, line2, line3, line4],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=4,
            frameon=False,
            prop={"family": "Arial", "size": 12},
        )

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")
            label.set_fontsize(12)

        if save:
            directory = "./Output/" + experiment + "/" + subexperiment + "/plots"
            os.makedirs(directory, exist_ok=True)

            name = network_name + "_" + "_".join([f"{k}={v}" for k, v in nx_params.items()])
            filename_base = (
                "/N={N}-T={T}-S={S}-expl={expl}-cg={cg}-cd={cd}-ts={ts}-ft={ft}-sw={sw}-post={post}-smooth={smooth}-{name}"
            ).format(
                N=self.N,
                T=self.T,
                S=self.S,
                expl=self.method_c_explainability,
                cg=self.controversy_growth,
                cd=self.controversy_decay,
                ts=self.trust_sensitivity,
                ft=self.fallback_threshold,
                sw=self.stable_window,
                post=self.post_stable_steps,
                smooth=self.plot_smoothing_window,
                name=name,
            )
            plt.savefig(directory + filename_base + ".png", bbox_inches="tight", dpi=600)

        plt.tight_layout()
        plt.show()
        plt.close()
