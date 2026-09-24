import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(str(Path(__file__).parents[1]))  # get to exp folder
BASE_DIR = Path(__file__).parent.parent  # covid_summer_outlook_2026/

plt.switch_backend("agg")
initial_date = pd.to_datetime("2022-02-11")

# Note: this is not the official HHS region definition; we omit PR, DC, and other territories
HHS_GROUPING = {
    1: ["CT", "ME", "MA", "NH", "RI", "VT"],
    2: ["NJ", "NY"],
    3: ["DE", "MD", "PA", "WV"],
    4: ["AL", "FL", "GA", "KY", "MS", "NC", "SC", "TN", "VA"],
    5: ["IL", "IN", "MI", "MN", "OH", "WI"],
    6: ["AR", "LA", "NM", "OK", "TX"],
    7: ["IA", "KS", "MO", "NE"],
    8: ["CO", "MT", "ND", "SD", "UT", "WY"],
    9: ["AZ", "CA", "HI", "NV"],
    10: ["AK", "ID", "OR", "WA"],
}


def retrieve_posterior_pred(state, az_output_path):
    json_file = os.path.join(az_output_path, state, "predictive_output.json")
    posterior_pred = json.load(open(json_file, "r"))
    # convert back to array
    for k, v in posterior_pred.items():
        if isinstance(v, list):
            posterior_pred[k] = np.array(v)

    return posterior_pred


def retrieve_fitted_medians(state, az_output_path):
    json_file = os.path.join(az_output_path, state, "samples.json")
    post_samp = json.load(open(json_file, "r"))
    fitted_medians = {
        k: jnp.median(jnp.array(v), axis=(0, 1)) for k, v in post_samp.items()
    }

    return fitted_medians


def calc_pred_quantiles(posterior_pred):
    data_dict = {
        "pred_hosps": posterior_pred["incidence"],
        "pred_hosps_all": jnp.sum(
            posterior_pred["incidence"], axis=2
        ),  # collapse age
        "pred_inf": posterior_pred["infection_incidence"],
        "pred_var": posterior_pred["variant_proportion"],
        "seasonality": posterior_pred["seasonality_mult"],
    }

    quantiles_dict = {}
    for k, v in data_dict.items():
        quantiles_dict[f"{k}_q05"] = np.quantile(v, 0.05, 0)
        quantiles_dict[f"{k}_q50"] = np.quantile(v, 0.50, 0)
        quantiles_dict[f"{k}_q95"] = np.quantile(v, 0.95, 0)

    return quantiles_dict


def fill_missing(sequence, values):
    seq_min = sequence.min()
    seq_max = sequence.max()

    fill = np.zeros(values[0].shape) * np.nan
    zip_dict = dict(zip(sequence, values))
    new_sequence = np.arange(seq_min, seq_max + 1)
    new_values = np.array(
        [zip_dict[x] if x in zip_dict.keys() else fill for x in new_sequence]
    )
    return new_sequence, new_values


def process_plot_state(state, az_output_path):
    posterior_pred = retrieve_posterior_pred(state, az_output_path)
    fitted_medians = retrieve_fitted_medians(state, az_output_path)
    fitted_medians["state"] = state
    median_df = pd.DataFrame(fitted_medians, index=[state])

    quantiles_dict = calc_pred_quantiles(posterior_pred)
    obs_hosps_age = posterior_pred["obs_hosps_weekly"]
    obs_hosps_all = np.sum(obs_hosps_age, axis=-1)  # collapse age
    obs_var_prop = posterior_pred["obs_var_prop"]
    num_strain = obs_var_prop.shape[-1]
    K = np.asarray(quantiles_dict["pred_var_q05"]).shape[
        -1
    ]  # uses model’s K (6)
    obs_var_prop = obs_var_prop[:, :K]  # match obs to model K

    week0_day = posterior_pred["week0_day"]
    # Filling in the gaps for hosps, if any
    obs_hosps_weeks = posterior_pred["obs_hosps_weeks"]
    obs_hosps_weeks, obs_hosps_all = fill_missing(
        obs_hosps_weeks, obs_hosps_all
    )
    # Make dates
    obs_hosps_dates = (obs_hosps_weeks * 7 + week0_day) * pd.Timedelta(
        days=1
    ) + initial_date
    obs_var_dates = (
        posterior_pred["obs_var_weeks"] * 7 + week0_day
    ) * pd.Timedelta(days=1) + initial_date
    sim_hosps_dates = (
        posterior_pred["hosps_weeks"] * 7 + week0_day
    ) * pd.Timedelta(days=1) + initial_date
    sim_var_dates = (
        posterior_pred["var_weeks"] * 7 + week0_day
    ) * pd.Timedelta(days=1) + initial_date

    # Calculate % of obs inside PI
    obs_hosps_coverage = np.mean(
        (
            obs_hosps_all
            > quantiles_dict["pred_hosps_all_q05"][0 : len(obs_hosps_all)]
        )
        & (
            obs_hosps_all
            < quantiles_dict["pred_hosps_all_q95"][0 : len(obs_hosps_all)]
        )
    )

    obs_var_coverage = np.mean(
        (obs_var_prop > quantiles_dict["pred_var_q05"][0 : len(obs_var_prop)])
        & (
            obs_var_prop
            < quantiles_dict["pred_var_q95"][0 : len(obs_var_prop)]
        )
    )
    obs_hosps_coverage = np.round(obs_hosps_coverage * 100, decimals=1)
    obs_var_coverage = np.round(obs_var_coverage * 100, decimals=1)

    # Plotting stuff
    date_major_freq = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    date_minor_freq = mdates.MonthLocator(interval=1)
    date_format = mdates.DateFormatter("%b\n%y")
    colors_age = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    colors_strain = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#1b9e77",
        "#d95f02",
        "#7570b3",
    ]
    fig, axs = plt.subplots(4, 1)
    # plt.minorticks_on()
    for ax in axs:
        ax.xaxis.set_major_locator(date_major_freq)
        ax.xaxis.set_minor_locator(date_minor_freq)
        ax.xaxis.set_major_formatter(date_format)

    # Configure plots
    # axs[0].set_prop_cycle(cycler(color=colors_age))
    axs[0].set_title(state + ": Observed vs fitted (Median +- 90% PI)")
    # axs[0].set_ylim([0.01, 500])
    # axs[0].set_yscale("log")
    axs[0].set_ylabel("Weekly hospitalization")

    # axs[1].set_yscale("log")
    axs[1].set_prop_cycle(cycler(color=colors_age))
    axs[1].set_ylabel("Weekly Infection")
    # axs[1].set_ylabel("Weekly hospitalization by age")
    # axs[1].set_yscale("log")

    axs[2].set_prop_cycle(cycler(color=colors_strain[:num_strain]))
    axs[2].set_ylabel("Variant proportion")

    axs[3].set_ylabel("Seasonality mult.")
    axs[3].set_prop_cycle(cycler(color=["#888888"]))

    # Hospitalization plot
    axs[0].plot(sim_hosps_dates, quantiles_dict["pred_hosps_all_q50"])
    axs[0].fill_between(
        sim_hosps_dates,
        quantiles_dict["pred_hosps_all_q05"],
        quantiles_dict["pred_hosps_all_q95"],
        alpha=0.5,
    )
    axs[0].plot(
        obs_hosps_dates,
        obs_hosps_all,
        c="#333333",
        linewidth=0.5,
        marker="o",
        markersize=1.5,
        alpha=0.6,
    )
    axs[0].text(
        0.99,
        0.99,
        f"Coverage: {str(obs_hosps_coverage)}%",
        transform=axs[0].transAxes,
        ha="right",
        va="top",
    )
    # Infection incidence plot
    axs[1].plot(sim_hosps_dates, quantiles_dict["pred_inf_q50"])
    axs[1].fill_between(
        sim_hosps_dates,
        quantiles_dict["pred_inf_q05"],
        quantiles_dict["pred_inf_q95"],
        alpha=0.5,
    )
    # axs[1].plot(sim_hosps_dates, quantiles_dict["pred_hosps_q50"])
    # for age in range(4):
    #     axs[1].fill_between(
    #         sim_hosps_dates,
    #         quantiles_dict["pred_hosps_q05"][:, age],
    #         quantiles_dict["pred_hosps_q95"][:, age],
    #         alpha=0.25,
    #     )
    # axs[1].plot(obs_hosps_dates, obs_hosps_age, linestyle=":")
    # Variant proportion plot
    axs[2].plot(sim_var_dates, quantiles_dict["pred_var_q50"])
    for lci, uci in zip(
        quantiles_dict["pred_var_q05"].T, quantiles_dict["pred_var_q95"].T
    ):
        axs[2].fill_between(sim_var_dates, lci, uci, alpha=0.2)
    for obs in posterior_pred["obs_var_prop"].T:
        axs[2].scatter(obs_var_dates, obs, s=4)
    axs[2].text(
        0.99,
        0.99,
        f"Coverage: {str(obs_var_coverage)}%",
        transform=axs[2].transAxes,
        ha="right",
        va="top",
    )
    # Seasonality multiplier plot
    axs[3].plot(sim_hosps_dates, quantiles_dict["seasonality_q50"])
    axs[3].fill_between(
        sim_hosps_dates,
        quantiles_dict["seasonality_q05"],
        quantiles_dict["seasonality_q95"],
        alpha=0.5,
    )

    axs[0].set_ylim(ymin=0.0)
    axs[1].set_ylim(ymin=0.0)
    fig.set_size_inches(8, 12)
    fig.set_dpi(300)

    return (fig, median_df)


def save_aggregated_posterior(states, name, job_path):
    agg_keys_sum = ["incidence", "infection_incidence", "obs_hosps_weekly"]
    agg_keys_mean = ["variant_proportion", "seasonality_mult", "obs_var_prop"]
    meta_keys = [
        "week0_day",
        "hosps_weeks",
        "obs_hosps_weeks",
        "var_weeks",
        "obs_var_weeks",
    ]

    agg_data = {key: [] for key in agg_keys_sum + agg_keys_mean}
    meta_data = {key: [] for key in meta_keys if key != "week0_day"}
    week0_day = None

    for state in states:
        json_path = os.path.join(job_path, state, "predictive_output.json")

        with open(json_path, "r") as f:
            pred = json.load(f)

        # Special handling of obs due to missing data
        obs_hosps_weeks = np.array(pred["obs_hosps_weeks"])
        obs_hosps_weekly = np.array(pred["obs_hosps_weekly"])
        obs_hosps_weeks, obs_hosps_weekly = fill_missing(
            obs_hosps_weeks, obs_hosps_weekly
        )
        pred["obs_hosps_weeks"] = obs_hosps_weeks.tolist()
        pred["obs_hosps_weekly"] = obs_hosps_weekly.tolist()

        for key in agg_keys_sum + agg_keys_mean:
            if key not in pred:
                continue

            arr = np.array(pred[key])
            if arr.ndim == 1:
                arr = arr[:, None]

            agg_data[key].append(arr)

        for mk in meta_keys:
            if mk not in pred:
                continue

            if mk == "week0_day":
                week0_day = pred[mk]
            else:
                meta_arr = np.array(pred[mk])
                meta_data[mk].append(meta_arr)

    # Padding arrays to the max length per key (for ragged aggregation)

    # TODO: Don't know what is this and this doesn't always do the thing...
    def pad_to_max(arr_list):
        if not arr_list:
            return []
        # Always convert arrays to float (if they are not already)
        arr_list = [a.astype(float) for a in arr_list]
        maxlen = max(a.shape[0] for a in arr_list)
        # arr_shape = arr_list[0].shape[1:] if arr_list[0].ndim > 1 else ()
        out = []
        for a in arr_list:
            pad_width = [(0, maxlen - a.shape[0])]
            if a.ndim > 1:
                pad_width += [(0, 0)] * (a.ndim - 1)
            out.append(np.pad(a, pad_width, constant_values=np.nan))
        return np.stack(out)

    combined = {}
    for key in agg_keys_sum:
        arrs = agg_data[key]
        if arrs:
            stacked = pad_to_max(arrs)
            all_nan = np.all(np.isnan(stacked), axis=0)
            nansum = np.nansum(stacked, axis=0)
            nansum[all_nan] = np.nan
            combined[key] = nansum.squeeze().tolist()

    for key in agg_keys_mean:
        arrs = agg_data[key]
        if arrs:
            stacked = pad_to_max(arrs)
            combined[key] = np.nanmean(stacked, axis=0).squeeze().tolist()

    if week0_day is not None:
        combined["week0_day"] = week0_day

    for mk in meta_data:
        arrs = meta_data[mk]
        if arrs:
            stacked = pad_to_max(arrs)
            combined[mk] = np.nanmean(stacked, axis=0).squeeze().tolist()

    agg_dir = os.path.join(job_path, name)
    os.makedirs(agg_dir, exist_ok=True)
    agg_json_path = os.path.join(agg_dir, "predictive_output.json")

    with open(agg_json_path, "w") as f:
        json.dump(combined, f)

    # Aggregate medians from all state samples.json files
    sample_dicts = []

    for state in states:
        samp_path = os.path.join(job_path, state, "samples.json")
        if not os.path.exists(samp_path):
            continue

        with open(samp_path, "r") as f:
            sample_dicts.append(json.load(f))

    median_agg = {}
    for key in sample_dicts[0]:
        try:
            arrays = [np.array(s[key]) for s in sample_dicts if key in s]
            if not arrays:
                continue

            # Pad to max length
            stacked = pad_to_max(arrays)
            median_agg[key] = np.nanmedian(stacked, axis=0).tolist()

        except Exception:
            continue

    agg_samples_path = os.path.join(agg_dir, "samples.json")
    with open(agg_samples_path, "w") as f:
        json.dump(median_agg, f)


def save_states_pdf_with_us(output_path, job_id):
    job_path = os.path.join(output_path, job_id)

    states = [
        d
        for d in os.listdir(job_path)
        if os.path.exists(os.path.join(job_path, d, "predictive_output.json"))
        and ((d != "US") and ("HHS" not in d))
    ]

    save_aggregated_posterior(states, "US", job_path)
    states_plus = ["US"]
    for i in range(1, 11):
        states_subset = HHS_GROUPING[i]
        intersect = np.intersect1d(np.array(states), np.array(states_subset))
        if intersect.size > 0:
            save_aggregated_posterior(intersect, f"HHS{i}", job_path)
            states_plus = states_plus + [f"HHS{i}"]

    states_plus = states_plus + states
    pool = mp.Pool(4)
    figs_and_dfs = pool.starmap(
        process_plot_state, [(st, job_path) for st in states_plus]
    )
    figs, median_dfs = zip(*figs_and_dfs)

    pdf_path = os.path.join(job_path, f"obs_vs_fitted_{job_id}.pdf")
    pdf_pages = PdfPages(pdf_path)

    for f in figs:
        pdf_pages.savefig(f, bbox_inches="tight")
        plt.close(f)
    pdf_pages.close()
    pd.concat(median_dfs).to_csv(
        os.path.join(job_path, "medians.csv"), index=False
    )
    pool.close()


parser = argparse.ArgumentParser(description="Experiment Azure Launcher")
parser.add_argument(
    "--job_id",
    "-j",
    type=str,
    help="job ID of the azure job, must be unique",
    required=True,
)

if __name__ == "__main__":
    args = parser.parse_args()
    job_id: str = args.job_id
    output_path = os.path.join(BASE_DIR, "output")
    save_states_pdf_with_us(output_path, job_id)
