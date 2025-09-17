# Data comparison

##### Setup #####

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/cifor")
       
library(tidyverse)
library(readxl)

source("./src/redd_int_funs.R")
source("./src/varlabs.R")

# load data
load("./data/hh_pd.RData")
load("./data/vil_pd.RData")


vdf <- read_xlsx("./data/Processed_GCS_village_panel_16Dec16.xlsx")

# import previously processed household data (for baseline matching)
prsd_hh_dat <- read_xlsx("./data/Processed_GCS_household_panel_16Dec16/Processed_GCS_household_panel_16Dec16.xlsx", 
                       sheet = 2, na = c("NA", "NaN"))



##### 1) Preprocessed vs. own controls #####



# collection of open questions

# why are there 283 household involved in the intervention in the baseline?
table(prsd_hh_dat$Period_desc, prsd_hh_dat$Redd_involved)

# why are there so many (60%) missing involvement values
table(is.na(prsd_hh_dat$Redd_involved))

prsd_hh_dat$Period <- ifelse(prsd_hh_dat$Period_desc == "before", 0, 1)

names(prsd_hh_dat)[10:ncol(prsd_hh_dat)-1] <- paste0("alt_", names(prsd_hh_dat)[10:ncol(prsd_hh_dat)-1])

# remove first 10 columns
prsd_hh_dat <- prsd_hh_dat[, c(1, 10:ncol(prsd_hh_dat))]

# merge data
ctrl_comp_dat <- hh_pd %>%
  left_join(prsd_hh_dat, by = c("Code_form", "Period"))

# make list of matching variable names from two datasets
ctrl_var_list <- list(c("hh_head_female", "alt_Gender"),              
                      c("hh_head_age" , "alt_Age"),              
                      c("hh_member", "alt_hh_size"),
                      c("alt_Year_formed", "Year_formed"),
                      # Assets
                      c("alt_floor_index", "Hh_floor2"),
                      c("alt_roof_index", "Hh_roof2"),
                      c("alt_wall_index", "Hh_walls2"),
                      c("alt_forest_income", "income_forest"),
                      c("alt_asset_val", "total_asset_value"),
                      # Land use
                      c("land_tot","alt_total_land"),
                      c("land_forest_tot","alt_for_land"),
                      c("land_ag_tot", "alt_ag_land"),
                      c("forest_share", "alt_forest_share"),     
                      c("ag_share", "alt_ag_share"))

lapply(ctrl_var_list, function(var){
  cat(paste(var[1], var[2], "\n"))
  corfun(ctrl_comp_dat, var[1], var[2])
})




##### 2) outcomes: preprocessing vs. own #####

# 3) outcomes: HH vs. Village

# aggregate self-reported data to village level
self_reported_vil_hh <- hh_pd %>%
  filter(Village %in% project_df$Village) %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018)) %>%
  group_by(year, Village) %>%
  summarise(sr_area_ha = sum(land_tot, na.rm = TRUE),
            sr_forest_ha = sum(land_forest_tot, na.rm = TRUE),
            sr_agric_ha = sum(land_ag_tot, na.rm = TRUE),
            sr_forest_perc = mean(forest_share, na.rm = TRUE),
            sr_ag_perc = mean(ag_share, na.rm = TRUE),
            sr_def_bin_perc = mean(Hh_clear, na.rm = TRUE),
            sr_def_ha = sum(hh_clearing_total, na.rm = TRUE),
            sf_def_perc = sr_def_ha / sr_area_ha) %>%
  ungroup()


### compare self reported outcomes based on aggregated HH vs. Village level
# vdf <- vdf %>%
#   mutate(year = as.integer(case_when(Period_desc == "before" ~ 2010,
#                             Period_desc == "after" ~ 2014)))
# 
# sf_hh_vil_comp <- left_join(self_reported_vil_hh, vdf, by = c("Village", "year")) %>%
#   filter(!is.na(sr_forest_perc),
#          !is.na(village_forest_per))
# 
# cor(sf_hh_vil_comp$sr_forest_perc, sf_hh_vil_comp$village_forest_per)

# -> weak correlation of 0.12, i do not trust the village level reported
# forest cover shares. Also because many shares are > 100, so not reliable.
# Instead, I use my own aggregated data based on HH level reports.

# Question is to what extent other village level variables are reliable.

# import village level forest cover data

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/cifor")

survey_path <- "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/1. Research/1. M2 Main Database/1. Global Database/"



p1.v.fc <- read_xls(paste0(survey_path, "1. Phase 1 Final/M2_GCS_REDD_PHASE1_VILLAGE_170601.xls"), sheet = 1) %>%
  filter(Country_code_desc %in% c("Brazil", "Indonesia", "Peru")) %>%
  mutate(year = 2010, 
         vsr_fc_perc = Forest_cover / Land_tot)

p2.v.fc <- read_xls(paste0(survey_path, "2. Phase 2 Final/M2_GCS_PHASE2_VILLAGE_170601.xls"), sheet = 1) %>%
  filter(Country_code_desc %in% c("Brazil", "Indonesia", "Peru")) %>%
  mutate(year = 2014,
         vsr_fc_perc = Forest_cover / Land_tot)

# merge phase 1 and 2
p1.2.v.fc <- bind_rows(p1.v.fc, p2.v.fc)

# p3.v.fc <- read_xlsx(paste0(survey_path, "7. Phase 3 Final/M2_GCS_PHASE3_VILLAGE_20210924.xlsx"), sheet = 1)# %>%
#   filter(Country_code_desc %in% c("Brazil", "Indonesia", "Peru")) %>%
#   mutate(year = 2014,
#          vsr_fc_perc = Forest_cover / Land_tot)


hist(p1.2.v.fc$vsr_fc_perc, breaks = 100, main = "Histogram of forest cover share", xlab = "Forest cover share", ylab = "Frequency")


# join with RS data
load("./data/rs_dat.RData")

sr_rs <- p1.2.v.fc %>%
  left_join(rs_dat, by = c("Village_name" = "Village", "year" = "year")) %>%
  filter(vsr_fc_perc >= 0 & vsr_fc_perc <= 1)

plot(sr_rs$vsr_fc_perc, sr_rs$jrc_perc_UndisturbedForest)

qqplot_fun(sr_rs, "vsr_fc_perc", "jrc_perc_UndisturbedForest", 
           title = "VSR vs. JRC: remaining forest (%)")

qqplot_fun(sr_rs %>% filter(Country_code=="101"), "vsr_fc_perc", "jrc_perc_UndisturbedForest", 
           title = "VSR vs. JRC: remaining forest (%)")

qqplot_fun(sr_rs %>% filter(Country_code=="102"), "vsr_fc_perc", "jrc_perc_UndisturbedForest", 
           title = "VSR vs. JRC: remaining forest (%)")

qqplot_fun(sr_rs %>% filter(Country_code=="300"), "vsr_fc_perc", "jrc_perc_UndisturbedForest", 
           title = "VSR vs. JRC: remaining forest (%)")


##### 4) outcomes: Self reported vs remotely sensed #####

# plot in loop for several outcomes

# prepare plotlist
plotlist.pts <- list()
plotlist.bplt <- list()

for (i in 1:length(outcomes_of_interest)) {
  p <- ggplot(vil_pd, aes(x = year, y = !!sym(outcomes_of_interest[i]))) +
    geom_point() +
    geom_smooth(method = "lm", aes(color=Village_Ty)) +
    # add title
    ggtitle(names(outcomes_of_interest)[i]) +
    #title(names(outcomes_of_interest)[i]) +
    labs(y = "", x = "") +
    theme_minimal()
  
  plotlist.pts[[i]] <- p
}

for (i in 1:length(outcomes_of_interest)) {
  p <- ggplot(vil_pd, aes(x = factor(year), y = !!sym(outcomes_of_interest[i]), fill = Village_Ty)) +
    geom_boxplot() +
    labs(y = i, x = "") +
    ggtitle(names(outcomes_of_interest)[i]) +
    theme_minimal()
  
  plotlist.bplt[[i]] <- p
}

# Combined plots
# - Deforestation rate (GFC, JRC, SR)
ggarrange(plotlist.bplt[[1]], plotlist.bplt[[6]], plotlist.bplt[[8]], 
  ncol = 1)

# - Forest cover (GFC, JRC, SR)
ggarrange(plotlist.bplt[[2]], plotlist.bplt[[3]], plotlist.bplt[[10]], 
          ncol = 1)


# correlation matrix
rs_corr_10 <- vil_pd %>%
  filter(year == 2010) %>%
  select(all_of(outcomes_of_interest)) %>%
  cor()

rs_corr_14 <- vil_pd %>%
  filter(year == 2014) %>%
  select(all_of(outcomes_of_interest)) %>%
  cor()

rs_corr_18 <- vil_pd %>%
  filter(year == 2018) %>%
  select(all_of(outcomes_of_interest)) %>%
  filter(complete.cases(.)) %>%
  cor()

corrplot(rs_corr_10, method = "number")
corrplot(rs_corr_14, method = "number")
corrplot(rs_corr_18, method = "number")




p.qq.fc <- qqplot_fun(vil_pd, "gfc_remaining_forest_perc", "jrc_perc_UndisturbedForest", 
                      title = "GFC vs. JRC: remaining forest (%)")

p.qq.def <- qqplot_fun(vil_pd, "gfc_def_perc_fc2000", "jrc_perc_DefTotal", 
                       title = "GFC vs. JRC: deforestation rate (%)")

# merge product comparison
ggarrange(p.qq.fc, p.qq.def, ncol=2)


p.qq.sr.fc <- qqplot_fun(vil_pd, 
                         "sr_forest_perc", "jrc_perc_UndisturbedForest", 
                         title = "Forest cover: Self-reported vs. JRC (%)")

p.qq.sr.fc2 <- qqplot_fun(vil_pd, 
                          "sr_forest_perc", "gfc_remaining_forest_perc", 
                          title = "Forest cover: Self-reported vs. GFC (%)")

p.qq.sr.fc.b <- qqplot_fun(vil_pd %>% filter(grepl("Brazil", project)), 
                           "sr_forest_perc", "jrc_perc_UndisturbedForest", 
                           title = "Forest cover: Self-reported vs. JRC (%)")

p.qq.sr.fc2.b <- qqplot_fun(vil_pd %>% filter(grepl("Brazil", project)), 
                            "sr_forest_perc", "gfc_remaining_forest_perc", 
                            title = "Forest cover: Self-reported vs. GFC (%)")

p.qq.sr.fc.o <- qqplot_fun(vil_pd %>% filter(!grepl("Brazil", project)), 
                           "sr_forest_perc", "jrc_perc_UndisturbedForest", 
                           title = "Forest cover: Self-reported vs. JRC (%)")

p.qq.sr.fc2.o <- qqplot_fun(vil_pd %>% filter(!grepl("Brazil", project)), 
                            "sr_forest_perc", "gfc_remaining_forest_perc", 
                            title = "Forest cover: Self-reported vs. GFC (%)")






p_fc_all <- ggarrange(p.qq.sr.fc, p.qq.sr.fc2, ncol = 2, common.legend = TRUE, 
                      legend = "bottom")

p_fc_brazil <- ggarrange(p.qq.sr.fc.b, p.qq.sr.fc2.b, ncol = 2, common.legend = TRUE, 
                         legend = "bottom")

p_fc_other <- ggarrange(p.qq.sr.fc.o, p.qq.sr.fc2.o, ncol = 2, common.legend = TRUE, 
                        legend = "bottom")

combine_plots <- ggarrange(p_fc_all, p_fc_brazil, p_fc_other, ncol = 1)

# save plots
ggsave("./out/fc_comparison.png", combine_plots, width = 22, height = 33, units = "cm", bg = "white")





