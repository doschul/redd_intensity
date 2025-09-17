# Simonet et al. 2018 reproduction



##### SETUP #####

# use transformed stata code provided in SI

# Load necessary libraries
library(dplyr)        # Data manipulation
library(MatchIt)      # Matching estimators
library(MASS)         # Probit model for Propensity Score Matching
library(stats)        # Summaries and regression

rm(list = ls())

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/cifor")


##### Original dataset #####


# Load dataset
#data <- read_csv("./data/simonet_2018/TD_Bresil/papier_DIDmatching_bresil_OK.csv")
data_ajae <- read_csv("./data/simonet_2018/TD_Bresil/papier_DIDmatching_bresil.csv")

# Generate unique ID
data_ajae <- data_ajae %>%
  mutate(id = row_number())

# Create treatment indicators
data_ajae <- data_ajae %>%
  mutate(
    D1 = ifelse(treated_vs_c == "1", 1, 0),
    D2 = ifelse(treated_vs_c == "1" | treated_vs_c == "NA", 1, 0),
    D3 = ifelse(treated_vs_c == "1" & treated_vs_c != "NA", 1, 0),
    D4 = ifelse(treated_vs_c == "NA", 1, 0)
  )

# Set weights based on treatment indicators
data_ajae <- data_ajae %>%
  mutate(
    weight = case_when(
      D1 == 1 ~ 0.1 / 0.49,
      D4 == 1 ~ 0.9 / 0.51,
      TRUE ~ 1
    )
  )

# Define control variables
X4 <- c("total_area_2010", "perc_forest_2008", "perc_forest_2010",
        "perc_cropAF_2010", "perc_pasture_2010", "total_value_2010",
        "stock_val_livestock_2010", "gvmt_bf_2010", "pension_2010",
        "salary_2010", "business_2010", "year_education", "age", "total_members")

# Control variables for placebo test
X5 <- X4[-2]  # Exclude "perc_forest_2008"

# Table 1 - Impact on participants in 2014
sum_list_ajae <- list(intervention = data_ajae %>%
                   filter(treatment ==1, !is.na(perc_pasture_2014)),
                 non_participants = data_ajae %>%
                   filter(is.na(D4)),
                 participants = data_ajae %>%
                   filter(D2 == 1, !is.na(perc_pasture_2014)),
                 comparison = data_ajae %>%
                   filter(D1 == 0, !is.na(perc_pasture_2014)))

lapply(sum_list_ajae, nrow)

# Summary statistics (Table 1)
lapply(sum_list_ajae, function(x){
  x %>%
    summarise(across(all_of(c(X4, "perc_forest_2014", "total_area_2014",
                              "perc_pasture_2014", "perc_cropAF_2014")), mean, .weights = weight))
}) %>% lapply(.,t) %>% do.call(cbind, .) %>% View()

# different first column, unclear why.


# Table 2 - Impact on participants in 2014 (Difference-in-Differences (DID))

# add change in outcomes (first difference)
data_ajae <- data_ajae %>%
  mutate(DY_totland = total_area_2014 - total_area_2010,
         DY_forest = perc_forest_2014 - perc_forest_2010,
         DY_agric = perc_cropAF_2014 - perc_cropAF_2010,
         DY_pasture = perc_pasture_2014 - perc_pasture_2010)

DV_2014 <- c("perc_forest_2014", "total_area_2014", "perc_pasture_2014", "perc_cropAF_2014")
DVs <- c("DY_forest", "DY_totland", "DY_agric", "DY_pasture")

# Outcome means in treated group
lapply(DV_2014, function(x){
  mean(data_ajae %>% 
         filter(D1 == 1) %>%
         pull(., x), na.rm = TRUE)
})

## Replicate Table 2, row 1

# DID Regression for all outcomes
lapply(DVs, function(x){
  #did_formula <- as.formula(paste(x, "~ D1 +", paste(X4, collapse = " + ")))
  did_formula <- as.formula(paste(x, "~ D1"))
  summary(lm(did_formula, data = data_ajae))
}) %>% lapply(., function(x) tidy(x)) %>%
  # filter D1 coefficient
  lapply(., function(x) x %>% filter(term == "D1")) %>% 
  do.call(rbind, .)

# run with control variables (last row)
lapply(DVs, function(x){
  did_formula <- as.formula(paste(x, "~ D1 +", paste(X4, collapse = " + ")))
  summary(lm(did_formula, data = data_ajae))
}) %>% lapply(., function(x) tidy(x)) %>%
  # filter D1 coefficient
  lapply(., function(x) x %>% filter(term == "D1")) %>% 
  do.call(rbind, .)





##### GCS panel dataset #####

load("./data/pd.RData")

pd <- pd %>%
  filter(!is.na(land_tot))

# identify households surveyed in baseline and first followup
wave_1_idx <- intersect(pd$Code_form[pd$Period == 0], 
                        pd$Code_form[pd$Period == 1])

wave_2_idx <- intersect(pd$Code_form[pd$Period == 0], 
                        pd$Code_form[pd$Period == 2])


# From Simonet et al. 2018
# A minimum of 30% of forest cover is required to be eligible for payments, 
# but only participants with at least 50% of forest cover can receive the full 
# amount. The project thus exceeds the requirements of the 2012 amendments to 
# the Brazilian Forest Code, which allow properties smaller than 400 hectares 
# to maintain only the forest cover they had in July 2008. The payments provide 
# an incentive to respect the old Forest Code in the ZEE (50% forest cover), 
# and thus sets targets that go beyond current legal requirements.

# select Transamazon data
tadat <- pd %>%
  filter(Country_code == "101", # select Brazil
         Project_code == "03",  # select Transamazon
         !is.na(land_tot)) %>%  # removes one missing outcome obs
  mutate(time = ifelse(Period == 0, 0, 1),
         treat = ifelse(Village_Type == 0, 0, 1), 
         did = time * treat)

# -> exclude households with less than 30% forest cover and larger than 400 ha
#tadat %>% filter(Period==0, forest_share < 0.3) %>% tally() # 14
#tadat %>% filter(Period==0, land_tot >= 400) %>% tally() # 2

# calculate difference between periods for forest share variabke
tadat1 <- tadat %>%
  filter(Code_form %in% wave_1_idx,
         Period < 2) %>%
  group_by(Code_form) %>%
  mutate(DY_forest = forest_share[Period == 1] - forest_share[Period == 0],
         DY_agric = ag_share[Period == 1] - ag_share[Period == 0],
         DY_totland = land_tot[Period == 1] - land_tot[Period == 0],
         PES = max(int_invol_bin_CLE)) %>%
  ungroup()


# replicate table 1 (SumStats)
tadat.filt.p0 <- tadat1 %>%
  filter(Period == 0) %>%
  filter(Code_form %in% wave_1_idx)


tadat.filt.p1 <- tadat1 %>%
  filter(Period == 1) %>%
  filter(Code_form %in% wave_1_idx)


sum_list_tadat.p0 <- list(intervention = tadat.filt.p0 %>%
                            filter(Village_Type == 1),
                          non_participants = tadat.filt.p0 %>%
                            filter(PES == 0 & Village_Type == 1),
                          participants = tadat.filt.p0 %>%
                            filter(PES == 1),
                          comparison = tadat.filt.p0 %>%
                            filter(Village_Type == 0))

lapply(sum_list_tadat.p0, nrow)

lapply(sum_list_tadat.p0, function(x){
  x %>%
    summarise(across(all_of(c( "forest_share", "land_tot")), mean))
}) %>% lapply(.,t) %>% do.call(cbind, .)



# main impact
non_participants <- tadat1 %>% 
  filter(PES == 0 & Village_Type == 1) %>%
  pull(Code_form)

repl_data <- tadat1 %>% filter(Period == 0, # to get baseline controls
                                       forest_share >= 0.3,
                                       !Code_form %in% non_participants)

# loop over outcomes
rep_dv <- c("DY_forest", "DY_agric", "DY_totland")

library(stargazer)

m_list <- lapply(rep_dv, function(x){
  f1 <- as.formula(paste(x, "~ PES"))
  m1 <- lm(f1, data = repl_data)
})

stargazer(m_list, type = "text")

# same with controls
ctrls <- c("land_tot", "land_forest_tot","total_asset_value",
           "hh_member", "hh_head_female", "hh_head_age")


m_list_ctrl <- lapply(rep_dv, function(x){
  f1 <- as.formula(paste(x, " ~ PES +", paste(ctrls, collapse = " + ")))
  m1 <- lm(f1, data = repl_data)
})

stargazer(m_list_ctrl, type = "text")

# --> looks similar to main results in Simonet et al. 2018

# check entire sample



rep_data_glob <- tadat1 %>% filter(Period == 0, # to get baseline controls
                                  forest_share >= 0.3)


##### Remotely sensed data #####

load("./data/rs_dat.RData")

rs_dat <- rs_dat %>%
  group_by(Village) %>%
  arrange(year) %>%
  # fill up all vars starting with int updown
  tidyr::fill(starts_with("int_"), .direction = "down") %>%
  # fill missing values in these columns with zero
  mutate(across(starts_with("int_"), ~replace_na(., 0))) %>%
  mutate(treat = ifelse(Village_Ty == "Intervention",1,0))

rs_ta <- rs_dat %>%
  filter(project == "Brazil_Transamazon")

# fix treatment to start in 2013
rs_ta$PES_any <- ((rs_ta$year >= 2013) * rs_ta$treat) * 1

table(rs_ta$PES_any, rs_ta$year)

# plot foest cover over time as lines colored by treatment status
rs_ta %>%
  filter(year >= 2008 & year <= 2015) %>%
  ggplot(aes(x = year, y = jrc_perc_UndisturbedForest,
             color = Village, linetype = Village_Ty)) +
  geom_line(lwd=1.2) +
  theme_minimal()


# change in RS forest cover between 2010 and 2014
rs_ta_dy <- rs_ta %>%
  group_by(Village) %>%
  mutate(DY_forest_10_14 = jrc_perc_UndisturbedForest[year == 2014] - jrc_perc_UndisturbedForest[year == 2010],
         DY_forest_11_14 = jrc_perc_UndisturbedForest[year == 2014] - jrc_perc_UndisturbedForest[year == 2011],
         DY_forest_12_14 = jrc_perc_UndisturbedForest[year == 2014] - jrc_perc_UndisturbedForest[year == 2012],
         DY_forest_13_14 = jrc_perc_UndisturbedForest[year == 2014] - jrc_perc_UndisturbedForest[year == 2013]) %>%
  ungroup() %>%
  # chose any post treatment year
  filter(year == 2014)

summary(lm(DY_forest_10_14 ~ PES_any, data = rs_ta_dy))
summary(lm(DY_forest_11_14 ~ PES_any, data = rs_ta_dy))
summary(lm(DY_forest_12_14 ~ PES_any, data = rs_ta_dy))
summary(lm(DY_forest_13_14 ~ PES_any, data = rs_ta_dy))

# use GFC data instead
# change in RS forest cover between 2010 and 2014
rs_ta_dy <- rs_ta %>%
  group_by(Village) %>%
  mutate(DY_forest_10_14 = gfc_remaining_forest_perc[year == 2014] - gfc_remaining_forest_perc[year == 2010],
         DY_forest_11_14 = gfc_remaining_forest_perc[year == 2014] - gfc_remaining_forest_perc[year == 2011],
         DY_forest_12_14 = gfc_remaining_forest_perc[year == 2014] - gfc_remaining_forest_perc[year == 2012],
         DY_forest_13_14 = gfc_remaining_forest_perc[year == 2014] - gfc_remaining_forest_perc[year == 2013]) %>%
  ungroup() %>%
  # chose any post treatment year
  filter(year == 2014)






library(DIDmultiplegtDYN)
# staggered diff in diff
did_multiplegt_dyn(df = rs_ta,
                   group = "Village",
                   time = "year",
                   treatment = "PES_any",
                   outcome = "jrc_perc_UndisturbedForest")


did_multiplegt_dyn(df = rs_ta, 
                   group = "Village",
                   time = "year",
                   treatment = "PES_any",
                   outcome = "jrc_perc_DefTotal")

# use entire bbrazi data
rs_bra <- rs_dat %>%
  filter(grepl("Brazil", project)) %>%
  mutate(PES_any = ((year >= 2013) * treat) * 1 ) %>%
  # sum up deforestation and degradation
  group_by(Village, year) %>%
  mutate(jrc_perc_DefDist = jrc_perc_DefTotal + jrc_perc_ForestDegradation)

table(rs_bra$PES_any, rs_bra$year)

did_multiplegt_dyn(df = rs_bra,
                   group = "Village",
                   time = "year",
                   treatment = "PES_any",
                   outcome = "jrc_perc_UndisturbedForest")

did_multiplegt_dyn(df = rs_bra,
                   group = "Village",
                   time = "year",
                   treatment = "PES_any",
                   outcome = "jrc_perc_DefTotal")
did_multiplegt_dyn(df = rs_bra,
                   group = "Village",
                   time = "year",
                   treatment = "PES_any",
                   outcome = "jrc_perc_ForestDegradation")

did_multiplegt_dyn(df = rs_bra,
                   group = "Village",
                   time = "year",
                   outcome = "jrc_perc_DefDist")

ta_p0 <- tadat1 %>%
  filter(Period == 0, 
         Code_form %in% wave_1_idx)

# change in RS forest cover between 2010 and 2014
# make crosssectional data
rs_bra_cs <- rs_bra %>%
  group_by(Village) %>%
  mutate(fc_2010 = jrc_perc_UndisturbedForest[year == 2010],
         DY_forest = jrc_perc_UndisturbedForest[year == 2014] - jrc_perc_UndisturbedForest[year == 2010],
         DY_DefDist = jrc_perc_DefDist[year == 2014] - jrc_perc_DefDist[year == 2010],
         # average deforestation rate between 2010 and 2014
         avg_def_10_14 = mean(jrc_perc_DirectDeforestation[year >=2010 & year <= 2014])) %>%
  ungroup() %>%
  # chose any post treatment year
  filter(year == 2014)

# quite different outcome data!
hist(simonet_main_data$DY_forest)
hist(rs_bra1$DY_forest)
hist(rs_bra_cs$avg_def_10_14)

summary(lm(DY_forest ~ PES_any + fc_2010, data = rs_bra_cs))
summary(lm(DY_DefDist ~ PES_any + fc_2010, data = rs_bra_cs))
summary(lm(avg_def_10_14 ~ PES_any + fc_2010, data = rs_bra_cs))
