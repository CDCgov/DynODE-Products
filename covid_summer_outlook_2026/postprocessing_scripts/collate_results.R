# Script to collate projection results from all states and scenarios in one
pacman::p_load(
  "progressr",
  "argparse",
  "yyjsonr",
  "arrow",
  "ggplot2",
  "dplyr",
  "data.table",
  "lubridate",
  "glue",
  "cowplot",
  "doFuture",
  "readr"
)
progressr::handlers(global = TRUE)
plan(multicore, workers = 5)

# Arg parsing
parser <- argparse::ArgumentParser(
  description = "Collate state-scenario results into two parquet files"
)
parser$add_argument("-j", "--jobid", required = TRUE, help = "Job ID")

args <- parser$parse_args()
job_id <- args$jobid

# Setting up
job_path <- file.path("./output", job_id)
states <- list.dirs(job_path, full.names = FALSE, recursive = FALSE)

# Looping through states and scenarios
loop_states <- function(states) {
  progress_bar <- progressr::progressor(along = states)
  foreach(
    st = states,
    .options.future = list(seed = TRUE)
  ) %dofuture%
    {
      progress_bar(sprintf("Working on state %s", st))
      state_dfs <- list()
      state_path <- file.path(job_path, st)
      scenarios <- list.dirs(
        state_path,
        full.names = FALSE,
        recursive = FALSE
      )
      for (s in scenarios) {
        out_json <- file.path(state_path, s, "predictive_output.json")
        js <- yyjsonr::read_json_file(out_json)
        week_dates <- js$week0_day +
          (js$hosps_weeks + 1) * 7 +
          ymd("2022-02-11")
        ### Adding removed week
        filter_start <- min(which(week_dates >= ymd("2024-03-01")))
        js$incidence <- js$incidence[filter_start:length(js$hosps_weeks), , ]
        js$hosps_weeks <- js$hosps_weeks[filter_start:length(js$hosps_weeks)]
        # Hospitalization
        incidence <- apply(js$incidence, 3, as.vector)
        # Unfold into a df
        weeks <- rep(
          js$hosps_weeks,
          dim(js$incidence)[2] * dim(js$incidence)[3]
        )

        # compute indices with intermediate names to keep line length < 80
        n_reps <- dim(js$incidence)[3]
        n_ages <- dim(js$incidence)[2]
        n_weeks <- length(js$hosps_weeks)
        id <- rep(
          seq_len(n_reps),
          each = n_weeks * n_ages
        )
        base_date <- ymd("2022-02-11")
        dates <- js$week0_day + (weeks + 1) * 7 + base_date
        ages <- rep(
          c("0-17", "18-49", "50-64", "65+"),
          each = length(js$hosps_weeks)
        ) |>
          rep(dim(js$incidence)[3])
        df <- tibble(
          date = dates,
          age = ages,
          hosp = as.vector(incidence),
          state = st,
          rep = id
        )
        state_dfs[[s]] <- df
      }
      # Observed data
      obs_dates <- js$week0_day +
        (js$obs_hosps_weeks + 1) * 7 +
        ymd("2022-02-11")
      obs_df <- data.frame(
        date = obs_dates,
        hosp = rowSums(js$obs_hosps_weekly),
        state = st
      )
      # Combined scenarios df for the state
      state_comb_df <- bind_rows(state_dfs, .id = "scenario")

      list(obs_df, state_comb_df)
    }
}

cat("Start looping\n")
output_list <- loop_states(states)

# save
allstates_sim_df <- allstates_obs_df <- list()
for (i in seq_along(output_list)) {
  allstates_obs_df[[i]] <- output_list[[i]][[1]]
  allstates_sim_df[[i]] <- output_list[[i]][[2]]
}

allstates_sim_df <- bind_rows(allstates_sim_df)
allstates_obs_df <- bind_rows(allstates_obs_df)

cat("Simulated dataframe shapes:")
print(dim(allstates_sim_df))
cat("Observed dataframe shapes:")
print(dim(allstates_obs_df))

arrow::write_parquet(
  allstates_sim_df,
  file.path(job_path, "projection_sim.parquet")
)
arrow::write_parquet(
  allstates_obs_df,
  file.path(job_path, "obs.parquet")
)
plan(sequential)
