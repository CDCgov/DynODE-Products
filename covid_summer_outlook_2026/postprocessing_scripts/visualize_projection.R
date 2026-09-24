pacman::p_load(
  "argparse",
  "arrow",
  "ggplot2",
  "dplyr",
  "data.table",
  "lubridate",
  "glue",
  "cowplot"
)
theme_set(theme_bw(base_size = 15))
theme_update(
  strip.background = element_blank(),
  panel.grid = element_line(linewidth = rel(0.25), colour = "#D1D3D4"),
  panel.grid.minor = element_blank(),
  panel.border = element_rect(linewidth = rel(0.5)),
  axis.ticks = element_line(linewidth = rel(0.25))
)

# Arg parsing
parser <- argparse::ArgumentParser(
  description = "Visualize projections after `collate_results`"
)
parser$add_argument("-j", "--jobid", required = TRUE, help = "Job ID")
args <- parser$parse_args()
jobid <- args$jobid

# Setting up
job_path <- file.path("./output", jobid)

# Loading data
allstates_sim_df <- arrow::read_parquet(file.path(
  job_path,
  "projection_sim.parquet"
))
allstates_obs_df <- arrow::read_parquet(file.path(
  job_path,
  "obs.parquet"
))

# Summarize dataframe and filter to period of interest
states_obs <- allstates_obs_df |>
  filter(date > ymd("2025-01-01"))
states_sim_summ <- allstates_sim_df |>
  group_by(scenario, date, state, rep) |>
  summarise(hosp = sum(hosp)) |>
  summarise(
    mean_hosp = mean(hosp),
    lci_hosp = quantile(hosp, 0.05),
    uci_hosp = quantile(hosp, 0.95)
  ) |>
  filter(date > ymd("2026-03-21"), date <= ymd("2026-09-30"))

# Visualize
ggplot(states_sim_summ) +
  geom_line(aes(x = date, y = mean_hosp, colour = scenario)) +
  geom_ribbon(
    aes(x = date, ymin = lci_hosp, ymax = uci_hosp, fill = scenario),
    alpha = 0.2,
    colour = NA
  ) +
  geom_line(aes(x = date, y = hosp), data = states_obs) +
  geom_point(
    aes(x = date, y = hosp),
    data = states_obs,
    color = "black",
    size = 2
  ) +
  facet_wrap(~state) +
  scale_colour_manual(
    values = c("firebrick", "navy"),
    labels = c(
      "With new major variant",
      "No new major variant"
    )
  ) +
  scale_fill_manual(
    values = c("firebrick", "navy"),
    labels = c(
      "With new major variant",
      "No new major variant"
    )
  ) +
  scale_x_date(date_breaks = "2 months", date_labels = "%b\n%y") +
  theme(
    legend.position = "inside",
    legend.justification = c(0.01, 0.99)
  ) +
  labs(x = "", y = "Weekly hospitalizations")
