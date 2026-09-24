####################################################
# Take CDC's variant proportion data, and reclassify
# the strains to the "major epoch" strains, and then
# recalculate the variant proportion by removing all
# the "others".
# Write out to 50-state csv based on HHS region
####################################################

pacman::p_load(dplyr, ggplot2, lubridate, data.table, tidyr, janitor)
theme_set(theme_bw())
theme_update(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold"),
  panel.grid = element_line(linewidth = rel(0.43), colour = "#D1D3D4"),
  panel.border = element_rect(linewidth = rel(0.43)),
  axis.ticks = element_line(linewidth = rel(0.43))
)

# start and end date
date_start <- ymd("2022-02-13")
date_end <- ymd("2026-02-14")
update_strain <- TRUE

# regions lookup
hhs <- c(
  4,
  10,
  9,
  6,
  9,
  8,
  1,
  3,
  4,
  4,
  9,
  10,
  5,
  5,
  7,
  7,
  4,
  6,
  1,
  3,
  1,
  5,
  5,
  4,
  7,
  8,
  7,
  9,
  1,
  2,
  6,
  2,
  4,
  8,
  5,
  6,
  10,
  3,
  1,
  4,
  8,
  4,
  6,
  8,
  1,
  4,
  10,
  3,
  5,
  8
)
hhs_lookup <- data.frame(state = state.abb, hhs = hhs)

# Process raw data taking only "empirical" data
# data source: https://data.cdc.gov/Laboratory-Surveillance/SARS-CoV-2-Variant-Proportions/jr58-6ysp #nolint
# Due to limitations, data is download as csv from the website with
# specific query such that published date is 2026-03-13. The csv is
# not included in this repository.
dat_csv <- "SARS-CoV-2_Variant_Proportions.csv"
dat <- data.table::fread(dat_csv) # Does not work since csv not included

dat_processed <- dat |>
  filter(
    modeltype == "empiric",
    share != 0
  ) |>
  mutate(date = as.Date(week_ending)) |>
  select(
    region = usa_or_hhsregion,
    date,
    variant,
    share
  )

# Subset to date of interests and only selecting variant with
# at least 1% appearance in its history
dates <- unique(dat_processed$date)
dat_subset <- dat_processed |>
  filter(date >= date_start, date <= date_end) |>
  group_by(variant) |>
  mutate(max_share = max(share)) |>
  filter(max_share > 0.01)

# Load variant lookup to reclassify variants to modeled strains
variants <- unique(dat_subset$variant) |> sort()
var_lookup <- fread(
  "./covid_summer_outlook_2026/data/variant-data/variant_lookup_2602_5s.csv"
)
setdiff(variants, var_lookup$variant) # All matched

# Reclassify variants to match current inference model
## Removing others (NA) from this
dat_reclass <- dat_subset |>
  left_join(var_lookup, by = "variant") |>
  filter(!is.na(strain)) |>
  group_by(region, date, strain) |>
  summarize(share = sum(share), .groups = "drop_last") |>
  mutate(share = share / sum(share)) |>
  ungroup()
## fill in the zeros
dat_reclass_fill <- dat_reclass |>
  tidyr::complete(
    region = unique(dat_reclass$region),
    date = unique(dat_reclass$date),
    strain = 0:4,
    fill = list(share = 0)
  )

# visualize
ggplot() +
  geom_area(
    aes(x = date, y = share, fill = as.factor(strain)),
    data = dat_reclass_fill
  ) +
  facet_wrap(~region)

# output by state
for (i in seq_along(state.abb)) {
  sta <- state.abb[i]
  stname <- state.name[i] |>
    stringr::str_replace_all(" ", "_")
  reg <- ifelse(
    sta == "US",
    "USA",
    hhs_lookup$hhs[hhs_lookup$state == sta] |> as.character()
  )

  outfile <- file.path(
    "./covid_summer_outlook_2026/data/variant-data/fitting-2602",
    glue("{stname}_strain_prop.csv")
  )

  df_out <- dat_reclass_fill |>
    filter(region == reg) |>
    arrange(date, strain)
  df_out |> data.table::fwrite(outfile)
}
