pacman::p_load(dplyr, lubridate, ggplot2, RSocrata, glue, stringr)

# Paths
nhsn_pre_2404_raw_path <- "/input/data/raw/NHSN_20200101_20240420.csv"
nhsn_pre_2404_processed_path <- "/input/data/processed/NHSN_20200101_20240420.csv"
nhsn_pos_2411_raw_path <- "/input/data/raw/NHSN_20241101.csv"
nhsn_pos_2411_processed_path <- "/input/data/processed/NHSN_20241101.csv"
fetch_new_pos_2411 <- TRUE

# Pre April 2024 NHSN data
if (!file.exists(nhsn_pre_2404_raw_path)) {
  df <- read.socrata(
    "https://healthdata.gov/resource/sgxm-t72h.json"
  ) # Retrieve data from healthdata if missing
  df <- df |>
    mutate(date = date |> as.Date())
  data.table::fwrite(df, nhsn_pre_2404_raw_path)
}
nhsn_pre_2404 <- data.table::fread(nhsn_pre_2404_raw_path)

## Pediatric
dat_pedia <- nhsn_pre_2404 |>
  select(
    state,
    date,
    previous_day_admission_pediatric_covid_confirmed
  ) |>
  group_by(state, date) |>
  summarise(
    new_admission = sum(
      previous_day_admission_pediatric_covid_confirmed,
      na.rm = TRUE
    )
  )
dat_pedia <- dat_pedia |>
  mutate(agegroup = "0-17")


## Adult
dat_adult <- nhsn_pre_2404 |>
  select(
    state,
    date,
    contains("previous_day_admission_adult_covid_confirmed")
  ) |>
  select(!contains("coverage")) |>
  group_by(state, date) |>
  summarise(across(everything(), \(x) sum(x, na.rm = TRUE)))

dat_adult_long <- dat_adult |>
  tidyr::pivot_longer(
    contains("previous_day_admission_adult_covid_confirmed"),
    names_to = "age",
    names_pattern = "previous_day_admission_adult_covid_confirmed_(.*)",
    values_to = "new_admission"
  ) |>
  filter(!is.na(age), age != "unknown") |>
  tidyr::separate(age, into = c("agelo", "agehi"), convert = TRUE) |>
  mutate(agehi = coalesce(agehi, 100))

### Because we want 50-64 age bin which cut through 60-69 in the hospitalization
### data, we assume that 60-69 is evenly distribute and we cut it in the middle
dat_60_69 <- dat_adult_long |>
  filter(agelo == 60)
dat_60_64 <- dat_60_69 |>
  mutate(
    agehi = 64,
    new_admission = new_admission / 2
  )
dat_65_69 <- dat_60_64 |>
  mutate(agelo = 65, agehi = 69)

dat_adult_long <- dat_adult_long |>
  filter(agelo != 60) |>
  bind_rows(dat_60_64, dat_65_69) |>
  arrange(date, agelo)

### Consolidating all the different age group into three
dat_adult_grouped <- dat_adult_long |>
  mutate(
    agegroup = case_when(
      agehi <= 49 ~ "18-49",
      agehi <= 64 ~ "50-64",
      TRUE ~ "65+"
    )
  ) |>
  group_by(state, date, agegroup) |>
  summarise(new_admission = sum(new_admission))

## Combined (pediatric plus adult)
dat_agegroup <- bind_rows(dat_pedia, dat_adult_grouped) |>
  arrange(date, agegroup)

### Aggregate to weekly data
dat_agegroup_subset <- dat_agegroup |>
  filter(state %in% state.abb) # Only 50 states

dat_agegroup_weekly <- dat_agegroup_subset |>
  mutate(week_end_date = grates::as_epiweek(date) |> grates::date_end()) |>
  group_by(date = week_end_date, state, agegroup) |>
  summarise(new_admission = sum(new_admission))

### Special handling of HI 0-17 age
# 0-17 becomes weird since Aug 2022, impute the rest of it using other agegroups
HI_agegroup_weekly <- dat_agegroup_weekly |> filter(state == "HI")
HI_training <- HI_agegroup_weekly |>
  filter(date >= ymd("2021-07-01"), date < ymd("2022-07-01"))
HI_conditionals <- HI_agegroup_weekly |>
  filter(date >= ymd("2022-07-01"), agegroup != "0-17")
mat <- matrix(HI_training$new_admission, nrow = 4) |> t()
m <- colMeans(mat)
sigma <- cov(mat)
sigma12 <- sigma[1, 2:4, drop = FALSE]
sigma22_inv <- solve(sigma[2:4, 2:4])
cond_mat <- matrix(HI_conditionals$new_admission, nrow = 3)

cond_m <- sigma12 %*% sigma22_inv %*% (cond_mat - m[2:4])
imputed_values <- round(as.vector(cond_m))
imputed_values <- ifelse(imputed_values < 0, 0, imputed_values)
cond <- dat_agegroup_weekly$date >= ymd("2022-07-01") &
  dat_agegroup_weekly$agegroup == "0-17" &
  dat_agegroup_weekly$state == "HI"
dat_agegroup_weekly[cond, ]$new_admission <- imputed_values

ggplot() +
  geom_line(
    aes(x = date, y = new_admission, colour = agegroup, group = agegroup),
    data = dat_agegroup_weekly |>
      filter(state == "HI")
  )

### Summing up all states to form US-wide data
dat_us <- dat_agegroup_weekly |>
  group_by(date, agegroup) |>
  summarise(new_admission = sum(new_admission)) |>
  mutate(state = "US")

## 50 states + US + Visualization
dat_all <- bind_rows(dat_agegroup_weekly, dat_us) |>
  ungroup() |>
  select(state, date, agegroup, hosp = new_admission)

dat_all |>
  ggplot() +
  geom_line(aes(x = date, y = hosp, colour = agegroup)) +
  theme_bw() +
  scale_y_log10() +
  facet_wrap(~state)

## Output
data.table::fwrite(dat_all, nhsn_pre_2404_processed_path)


# Post Nov 2024 NHSN data
if (!file.exists(nhsn_pos_2411_raw_path) || fetch_new_pos_2411) {
  df <- read.socrata("https://data.cdc.gov/resource/ua7e-t2fy.json")
  df <- df |>
    mutate(weekendingdate = weekendingdate |> as.Date())
  data.table::fwrite(df, nhsn_pos_2411_raw_path)
  maxdate <- max(df$weekendingdate)
  cat(glue(
    "Downloaded new NHSN data, which is updated to the week of {maxdate}"
  ))
}

nhsn_pos_2411 <- data.table::fread(nhsn_pos_2411_raw_path)
dat <- nhsn_pos_2411 |>
  rename(
    date = weekendingdate,
    state = jurisdiction
  ) |>
  filter(date >= ymd("2024-11-01")) |> # Pre Nov data not needed here
  select(date, state, matches("numconfc19newadm(adult|ped)")) |>
  select(!ends_with("per100k")) |>
  select(!ends_with("pctchg"))

dat_long <- dat |>
  tidyr::pivot_longer(
    -c(date, state),
    names_to = "agegroup",
    names_prefix = "numconfc19newadm(adult|ped)",
    values_to = "hosp"
  )

dat_agegroup <- dat_long |>
  tidyr::separate(
    agegroup,
    c("agemin", "agemax"),
    sep = "(to|plus)",
    convert = TRUE
  ) |>
  mutate(
    agegroup = case_when(
      agemin < 18 ~ "0-17",
      agemin < 50 ~ "18-49",
      agemin < 65 ~ "50-64",
      agemin < 100 ~ "65+"
    )
  ) |>
  group_by(date, state, agegroup) |>
  summarise(hosp = sum(hosp))

## Take 50 states + US (reaggregated)
dat_agegroup_subset <- dat_agegroup |>
  filter(state %in% state.abb)

dat_us <- dat_agegroup_subset |>
  group_by(date, agegroup) |>
  summarise(hosp = sum(hosp)) |>
  mutate(state = "US")

dat_all <- bind_rows(dat_agegroup_subset, dat_us)

## Output
data.table::fwrite(dat_all, nhsn_pos_2411_processed_path)

table(df$jurisdiction)
df |>
  filter(
    jurisdiction %in% c(state.abb, "USA"),
    weekendingdate >= ymd("2022-11-01")
  ) |>
  ggplot() +
  geom_line(aes(
    x = weekendingdate,
    y = as.numeric(totalconfc19newadmperchosprep)
  )) +
  facet_wrap(~jurisdiction)

dat_us |>
  group_by(date) |>
  summarise(hosp = sum(hosp)) |>
  ggplot() +
  geom_line(aes(date, hosp))
