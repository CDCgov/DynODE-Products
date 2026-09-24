pacman::p_load(dplyr, ggplot2, lubridate, tidyr, glue, data.table)
theme_set(theme_bw(base_size = 13))
theme_update(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.3), colour = "#D1D3D4"),
  panel.grid.minor = element_blank(),
  panel.border = element_rect(linewidth = rel(0.5)),
  axis.ticks = element_line(linewidth = rel(0.3))
)

# Paths and stuff
processed_path <- "/input/data/processed"
nhsn_pre_2404_processed_path <- file.path(
  processed_path,
  "NHSN_20200101_20240420.csv"
)
nhsn_pos_2411_processed_path <- file.path(processed_path, "NHSN_20241101.csv")
nssp_inferred_nhsn_path <- file.path(
  processed_path,
  "NSSP_inferred_NHSN_20240427_20260404.csv"
)
output_folder <- "/input/data/hospitalization-data/20220219_to_20260404/"
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Construct hospitalization df
nhsn_pre_2404 <- fread(nhsn_pre_2404_processed_path)
nhsn_pos_2411 <- fread(nhsn_pos_2411_processed_path)
nssp_inferred_nhsn <- fread(nssp_inferred_nhsn_path)

# switch NSSP to bi-weekly during blackout
# biweekly_end_date <- c(seq(as.Date("2024-05-04"), as.Date("2024-10-19"), "2 week"), as.Date("2024-10-26"))
# blackout_dates <- seq(as.Date("2024-04-27"), as.Date("2024-10-26"), "1 week")
# ind <- findInterval(blackout_dates, biweekly_end_date, left.open = TRUE) + 1
# biweekly_conversion <- data.frame(week_end_date = blackout_dates, biweek_end_date = biweekly_end_date[ind])
# nssp_inferred_nhsn_mod <- nssp_inferred_nhsn |>
#   mutate(week_end_date = as.Date(week_end_date)) |>
#   left_join(biweekly_conversion) |>
#   mutate(biweek_end_date = coalesce(biweek_end_date, week_end_date))
# nssp_inferred_nhsn_mod <- nssp_inferred_nhsn_mod |>
#   group_by(state, agegroup, biweek_end_date) |>
#   summarise(hosp = mean(hosp)) |>
#   mutate(biweek_end_date = as.IDate(biweek_end_date))
nssp_inferred_nhsn_mod <- nssp_inferred_nhsn

us_df_list <- list()
for (i in seq_along(state.name)) {
  stn <- state.name[i]
  stn1 <- str_replace(stn, " ", "_")
  sta <- state.abb[i]

  nhsn_pre_2404_st <- nhsn_pre_2404 |>
    filter(state == sta, date > ymd("2022-02-11"))

  if (stn == "Wyoming") {
    # Wyoming: use only NHSN pre-2404 data
    nhsn_pos_2411_st <- nhsn_pos_2411 |>
      filter(state == sta)

    full_hosp_series <- bind_rows(
      nhsn_pre_2404_st,
      nhsn_pos_2411_st
      # no nssp_st
    ) |>
      arrange(date)
  } else {
    # Switch dates for all other states (Missouri included)
    if (stn %in% c("Pennsylvania", "Arizona")) {
      switch_date <- ymd("2025-05-07")
    } else {
      switch_date <- ymd("2024-12-21")
    }

    nssp_st <- nssp_inferred_nhsn_mod |>
      filter(state == sta, week_end_date <= switch_date) |>
      rename(date = week_end_date)

    nhsn_pos_2411_st <- nhsn_pos_2411 |>
      filter(state == sta, date > switch_date)

    full_hosp_series <- bind_rows(
      nhsn_pre_2404_st,
      nhsn_pos_2411_st,
      nssp_st
    ) |>
      arrange(date)
  }

  full_hosp_series <- full_hosp_series |>
    mutate(hosp = hosp / 7) # This is for DynODE input (to account for incomplete week)

  outpath <- file.path(output_folder, glue("{stn1}_hospitalization.csv"))
  fwrite(full_hosp_series, outpath)
  us_df_list[[stn1]] <- full_hosp_series
}

us_df <- bind_rows(us_df_list)
us_df |>
  group_by(date) |>
  summarise(hosp = sum(hosp)) |>
  filter(date >= ymd("2024-01-01")) |>
  ggplot() +
  geom_line(aes(date, hosp))
