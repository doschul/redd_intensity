# make summary statistics and other descriptive data insights.

##### Setup #####

rm(list = ls())

# libraries
library(tidyverse)
library(gtsummary)
library(gt)

# load data
source("./src/redd_int_funs.R")
source("./src/varlabs.R")
load("./data/hh_pd_bal.RData")
load("./data/int_df_full.RData")
load("./data/vil_pd.RData")

int_df_full$Period <- NULL

hh_pd_bal <- hh_pd_bal %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018))

# restrict to baseline forest
pd <- hh_pd_bal %>%
  #filter(Code_form %in% hh_baseline_with_forest) %>%
  left_join(., int_df_full, by = c("Code_form", "year")) %>%
  filter(!is.na(int_cat_tot_avail))

# make forest change variables, short and long term
pd <- pd %>%
  group_by(Code_form) %>%
  mutate(forest_change_2014 = forest_share[year == 2014] - forest_share[year == 2010],
         forest_change_2018 = forest_share[year == 2018] - forest_share[year == 2010],
         clearing_any_2014 = Hh_clear[year == 2014],
         clearing_any_2018 = Hh_clear[year == 2018],
         clearing_ha_2014 = hh_clearing_total[year == 2014],
         clearing_ha_2018 = hh_clearing_total[year == 2018],
  ) %>%
  ungroup()

##### Summary statistics #####

#### Household ####

sumvar_hh_labels <- list(hh_head_female = "HH head female (1=yes)",
                         hh_head_occup_agric = "Agricultural occupation (1=yes)",
                         hh_head_age = "HH head age",
                         hh_member = "HH members",
                         hh_dependency_ratio = "Dependency ratio",
                         Hh_floor2 = "Floor material",
                         Hh_walls2 = "Wall material",
                         Hh_roof2 = "Roof material",
                         Hh_water = "Water source",
                         Hh_toilet = "Toilet type",
                         Hh_electric = "Electricity",
                         Hh_cooking_fuel = "Cooking fuel",
                         land_tot = "Total land owned (ha)",
                         forest_share = "Forest share (%)",
                         forest_change_2014 = "Forest change 2014 (%)",
                         forest_change_2018 = "Forest change 2018 (%)",
                         clearing_any_2014 = "Clearing 2014 (1=yes)",
                         clearing_any_2018 = "Clearing 2018 (1=yes)",
                         clearing_ha_2014 = "Clearing 2014 (ha)",
                         clearing_ha_2018 = "Clearing 2018 (ha)"
                         )

# subset data to baseline
sumdat_hh_bl <- pd %>% 
  filter(year == 2010) %>%
  mutate(Village_Type_descr = case_when(Village_Type == 1 ~ "Treatment",
                                        Village_Type == 0 ~ "Control"))
  
# fill missing values with village mean
sumdat_hh_bl <- sumdat_hh_bl %>%
  group_by(Village) %>%
  mutate_at(vars(all_of(sumvars_hh)), ~replace_na(., mean(., na.rm = TRUE)))  

sumstat.hh.bl <- sumdat_hh_bl %>%
  tbl_summary(by = Village_Type_descr, 
              include = all_of(names(sumvar_hh_labels)),
              label = sumvar_hh_labels,
              type = list(names(sumvar_hh_labels)[1:2] ~"dichotomous",
                          names(sumvar_hh_labels)[3:length(names(sumvar_hh_labels))] ~ "continuous"),
              statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~
                                 "{n} ({p}%)"),) %>%
  add_overall() %>%
  add_p() %>%
  as_gt()

# save

gtsave(sumstat.hh.bl, file = "./out/tbl/sumstat_hh_bl.html")

#### Village ####

# aggregate some variables across time
vil_pd_agg <- vil_pd %>%
  filter(!is.na(n_vil)) %>%
  mutate(jrc_fc2000_perc = jrc_fc2000_ha / area_ha * 100) %>%
  group_by(Village_Ty, Village) %>%
  summarise(n_vil = unique(n_vil),
            area_ha = unique(area_ha),
            jrc_fc2000_perc = unique(jrc_fc2000_perc),
            tot_mean_prec = mean(tot_mean_prec),
            tot_mean_tmmx = mean(tot_mean_tmmx),
            sr_forest_perc = mean(sr_forest_perc[year %in% c(2010, 2014, 2018)]) * 100,
            sf_def_perc = mean(sf_def_perc[year %in% c(2010, 2014, 2018)]) * 100,
            jrc_perc_UndisturbedForest = mean(jrc_perc_UndisturbedForest) * 100,
            jrc_perc_DirectDeforestation = mean(jrc_perc_DirectDeforestation) * 100,
            jrc_perc_ForestDegradation = mean(jrc_perc_ForestDegradation) * 100,
            jrc_perc_DeforAfterDegrad = mean(jrc_perc_DeforAfterDegrad) * 100) %>%
  ungroup()

sumvar_vil_labels <- list(n_vil = "Number of sampled households",
                          area_ha = "Total village area (ha)",
                          tot_mean_prec = "Mean annual precipitation",
                          tot_mean_tmmx = "Mean annual maximum temperature",
                          # self-reported forest cover
                          sr_forest_perc = "Self-reported forest cover (%)",
                          sf_def_perc = "Self-reported deforestation (%)",
                          # JRC forest cover
                          jrc_fc2000_perc = "Forest cover 2000 (%)",
                          jrc_perc_UndisturbedForest = "JRC undisturbed forest (%)",
                          jrc_perc_DirectDeforestation = "JRC direct deforestation (%)",
                          jrc_perc_ForestDegradation = "JRC forest degradation (%)",
                          jrc_perc_DeforAfterDegrad = "JRC deforestation after degradation (%)")
                          

sumstat.vil <- vil_pd_agg %>%
  tbl_summary(by = Village_Ty, 
              include = all_of(names(sumvar_vil_labels)),
              label = sumvar_vil_labels,
              type = list(everything() ~ "continuous"),
              statistic = list(all_continuous() ~ "{mean} ({sd})")) %>%
  add_overall() %>%
  add_p() %>%
  as_gt()

gtsave(sumstat.vil, file = "./out/tbl/sumstat_vil.html")


#### Treatment distribution ####

trt_var_allint <- list(int_cat_tot_availALL = "Total available", 
                       int_cat_pos_availALL = "Positive available", 
                       int_cat_ena_availALL = "Enabling available", 
                       int_cat_neg_availALL = "Negative available", 
                       int_cat_tot_involALL = "Total involved", 
                       int_cat_pos_involALL = "Positive involved",
                       int_cat_ena_involALL = "Enabling involved",
                       int_cat_neg_involALL = "Negative involved",
                       int_cat_tot_chngeALL = "Total changed behavior", 
                       int_cat_pos_chngeALL = "Positive changed behavior",
                       int_cat_ena_chngeALL = "Enabling changed behavior", 
                       int_cat_neg_chngeALL = "Negative changed behavior")

trt_var_REDD <- list(int_cat_tot_avail = "Total available", 
                     int_cat_pos_avail = "Positive available", 
                     int_cat_ena_avail = "Enabling available", 
                     int_cat_neg_avail = "Negative available", 
                     int_cat_tot_invol = "Total involved", 
                     int_cat_pos_invol = "Positive involved", 
                     int_cat_ena_invol = "Enabling involved", 
                     int_cat_neg_invol = "Negative involved", 
                     int_cat_tot_chnge = "Total changed behavior", 
                     int_cat_pos_chnge = "Positive changed behavior", 
                     int_cat_ena_chnge = "Enabling changed behavior", 
                     int_cat_neg_chnge = "Negative changed behavior")

trt_hh_ALL <- int_df_full %>%
  filter(Code_form %in% pd$Code_form) %>%
  filter(year==2014) %>%
  tbl_summary(include = all_of(names(trt_var_allint)),
              label = trt_var_allint,
              type = list(names(trt_var_allint) ~ "categorical"),
              statistic = list(all_categorical() ~ "{n} ({p}%)"))

trt_hh_REDD <- int_df_full %>%
  filter(Code_form %in% pd$Code_form) %>%
  filter(year==2014) %>%
  # remove ALL columns to match column names of trt_hh_ALL
  select(-contains("ALL")) %>%
  # add "ALL" to column names to match column names of trt_hh_ALL
  rename_with(~paste0(., "ALL")) %>%
  tbl_summary(include = all_of(names(trt_var_allint)),
              label = trt_var_allint,
              type = list(names(trt_var_allint) ~ "categorical"),
              statistic = list(all_categorical() ~ "{n} ({p}%)"))
  

# merge tables
sum_trt_hh <- gtsummary::tbl_merge(list(trt_hh_ALL, trt_hh_REDD), 
                     tab_spanner = c("**All**", "**REDD+ only**")) %>%
  as_gt()

gtsave(sum_trt_hh, file = "./out/tbl/sum_trt_hh.html")

# Village level
trt_vil_ALL <- vil_pd %>%
  filter(!is.na(int_cat_tot_avail)) %>%
  filter(year==2014) %>%
  tbl_summary(include = all_of(names(trt_var_allint)),
              label = trt_var_allint,
              type = list(names(trt_var_allint) ~ "continuous"),
              statistic = list(all_continuous() ~ "{mean} ({sd})"))

trt_vil_REDD <- vil_pd %>%
  filter(year==2014) %>%
  filter(!is.na(int_cat_tot_avail)) %>%
  # remove ALL columns to match column names of trt_hh_ALL
  select(-contains("ALL")) %>%
  # add "ALL" to column names to match column names of trt_hh_ALL
  rename_with(~paste0(., "ALL")) %>%
  tbl_summary(include = all_of(names(trt_var_allint)),
              label = trt_var_allint,
              type = list(names(trt_var_allint) ~ "continuous"),
              statistic = list(all_continuous() ~ "{mean} ({sd})"))

# merge tables
sum_trt_vil <-gtsummary::tbl_merge(list(trt_vil_ALL, trt_vil_REDD), 
                     tab_spanner = c("**All**", "**REDD+ only**")) %>%
  as_gt()

# save
gtsave(sum_trt_vil, file = "./out/tbl/sum_trt_vil.html")

##### Data insights #####



vil_trt_plt_neg_invol <- ggplot(vil_pd %>% filter(!is.na(int_cat_neg_invol)), 
                                aes(x = year, 
                                    y = Village, 
                                    fill = int_cat_neg_invol)) +
  geom_raster() +
  theme_minimal()

vil_trt_plt_neg_involALL <- ggplot(vil_pd %>% filter(!is.na(int_cat_neg_invol)),  
                                   aes(x = year, 
                                       y = Village, 
                                       fill = int_cat_neg_involALL)) +
  geom_raster() +
  theme_minimal() +
  # remove axis labels
  theme(axis.text.y = element_blank(),
        axis.title.y = element_blank())

# combine the plots
library(ggpubr)
ggarrange(vil_trt_plt_neg_invol, vil_trt_plt_neg_involALL, ncol = 2, 
          common.legend = TRUE, legend = "bottom", widths = c(3, 2.1))


# correlation of REDD and nonREDD total treatment
plot(int_df_full$int_cat_tot_invol, int_df_full$int_nonREDD_ongoing_sum, 
     xlab = "REDD", ylab = "nonREDD", main = "Correlation of REDD and nonREDD total treatment")

# make grid density plot
ggplot(int_df_full, aes(x = int_cat_tot_invol, y = int_nonREDD_ongoing_sum)) +
  geom_bin2d() +
  # scale fill gradient
  scale_fill_viridis_c(trans = "log") 


# make grid density plot
ggplot(vil_pd, aes(x = int_cat_tot_invol, y = int_nonREDD_ongoing_sum)) +
  geom_bin2d() +
  # scale fill gradient
  scale_fill_viridis_c(trans = "log") 

# make grid density plot
ggplot(vil_pd %>% filter(project == "Brazil_Transamazon"), aes(x = int_cat_tot_invol, y = int_nonREDD_ongoing_sum)) +
  geom_bin2d() +
  # scale fill gradient
  scale_fill_viridis_c(trans = "log") 


