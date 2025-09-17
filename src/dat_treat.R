# Purpose: cleaning treatment data for household and village level
# Last update: 12.11.2024

##### Setup #####

rm(list=ls())

library(tidyverse)
library(readxl)

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/redd_intensity")
datapath <- "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/1. Research/1. M2 Main Database/1. Global Database"

# settings
years_of_analysis <- 2000:2020
intervention_type_short <- c("CLE", "EE", "FE", "NCLE", "RFAC", "TC", "OI")

# load variable labels
source("./src/varlabs.R")
source("./src/redd_int_funs.R")

##### Load and clean treatment data #####

# load SVI data
svi <- read_xlsx("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/1. Research/3. M2 Other Survey Data/2. Compiled Data/06. Survey of Village Intervention (SVI)/SVI Phase 2/SVI - Compiled 2016_04_25.xlsx", 
                 sheet = 1, na = c("", "-8", "-9", "N/A", "NA"))

# fix one misspelled village name
svi$Village_name[svi$Village_name == "Van Minh, Gia Vien Commune"] <- "Van Minh,  Gia Vien Commune"

# define columns to replace missin values
na_cols <- c("Done_by\r\n1=proponent\r\n2=affil.org\r\n3=both\r\n4=neither",
             "REDD+_strategy\r\n1=yes\r\n2=no", "REDD+_ongoing\r\n1=yes\r\n2=no", 
             "Included_in_HH_survey \r\n(1=yes, 0=no)")

# rename these columns
names(svi)[names(svi) %in% na_cols] <- c("proponent", "strategy", "ongoing", "included")

svi <- svi %>%
  # replace NA by zero in selected columns
  mutate(across(c("proponent", "strategy", "ongoing", "included"), ~replace_na(., 0))) %>%
  mutate(isREDD = ifelse(((proponent %in% c(1, 3)) & (strategy == 1) & (ongoing == 1)), 1, 0))

p2.treat <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                     sheet = "Tbl_Household_5B", col_types = "text")
p2.treat$Yes_no_unrated[is.na(p2.treat$Yes_no_unrated)] <- "N"
#p2.treat$Redd_involved[is.na(p2.treat$Redd_involved)] <- 0

# merge with svi data
p2.treat <- left_join(p2.treat, svi, by = c("Village" = "Village_name", 
                                            "Intervention_name_code" = "Code_of_intervention"))

# make intervention abbreviations
p2.treat <- p2.treat %>%
  mutate(intervention_type_short = case_when(
    Intervention_type_desc == "conditional livelihood enhancement" ~ "CLE",
    Intervention_type_desc == "environmental education" ~ "EE",
    Intervention_type_desc == "forest enhancement" ~ "FE",
    Intervention_type_desc == "non-conditional livelihood enhancement" ~ "NCLE",
    Intervention_type_desc == "restriction on forest access & conversion" ~ "RFAC",
    Intervention_type_desc == "tenure clarification" ~ "TC",
    Intervention_type_desc == "others intervention" ~ "OI",
    TRUE ~ "NA"
  ))

p2.treat$isREDD_descr <- ifelse(p2.treat$isREDD == 1, "REDD+ interventions", "nonREDD+ interventions")
table(p2.treat$Intervention_type_desc, p2.treat$Country, p2.treat$isREDD_descr)

##### Create HH-treatment variables #####

# loop over years and interventions
int_df_p2 <- data.frame()

p2.treat.wide.nonREDD <- p2.treat %>%
  filter(isREDD == 0) %>%
  group_by(Code_form) %>%
  summarise(Period = 1,
            nonREDD_int_ongoing_sum = sum(ongoing == 1),
            nonREDD_int_ended_sum = sum(ongoing == 0)) %>%
  ungroup()

# apply function to different data subsets
int_df_p2_nonREDD_o <- int_loop_fun(p2.treat %>% 
                                      filter(isREDD == 0, ongoing == 1), 
                                    years = years_of_analysis,
                                    intervention_type_short = intervention_type_short,
                               expo_prefix = "nonREDD_ongoing")
int_df_p2_nonREDD_e <- int_loop_fun(p2.treat %>% 
                                      filter(isREDD == 0, ongoing == 0), 
                                    years = years_of_analysis,
                                    intervention_type_short = intervention_type_short,
                                  expo_prefix = "nonREDD_ended")

int_df_nonREDD <- expand.grid(Code_form = unique(p2.treat$Code_form),
                              year = years_of_analysis) %>%
  left_join(., int_df_p2_nonREDD_o) %>%
  left_join(., int_df_p2_nonREDD_e)

int_df_nonREDD[["int_nonREDD_ongoing_sum"]] <- apply(
  int_df_nonREDD[,which(grepl("ongoing_sum", names(int_df_nonREDD)))], 1, sum
  )

int_df_nonREDD[["int_nonREDD_ended_sum"]] <- apply(
  int_df_nonREDD[,which(grepl("ended_sum", names(int_df_nonREDD)))], 1, sum
  )

int_df_nonREDD <- int_df_nonREDD %>%
  select(Code_form, year, int_nonREDD_ongoing_sum, int_nonREDD_ended_sum)

int_df_p2_redd_avail <- int_loop_fun(p2.treat %>% filter(isREDD == 1), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "avail")

int_df_p2_redd_invol <- int_loop_fun(p2.treat %>% filter(isREDD == 1, 
                                                    Hh_involved == 1), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "invol")

int_df_p2_redd_chnge <- int_loop_fun(p2.treat %>% filter(isREDD == 1, 
                                                    Yes_no_unrated == "Y"), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "chnge")

# make intesity indicators irrespective of REDD status (count all interventions!)
int_df_p2_all_avail <- int_loop_fun(p2.treat %>% filter(ongoing == 1), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "availALL")

int_df_p2_all_invol <- int_loop_fun(p2.treat %>% filter(ongoing == 1,
                                                        Hh_involved == 1), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "involALL")

int_df_p2_all_chnge <- int_loop_fun(p2.treat %>% filter(ongoing == 1,
                                                        Yes_no_unrated == "Y"), 
                                intervention_type_short = intervention_type_short,
                                years = years_of_analysis,
                                expo_prefix = "chngeALL")

# left join data on year and codeform
int_df_p2 <- expand.grid(Code_form = unique(p2.treat$Code_form),
                         year = years_of_analysis) %>%
  left_join(., int_df_nonREDD) %>%
  left_join(., int_df_p2_redd_avail) %>%
  left_join(., int_df_p2_redd_invol) %>%
  left_join(., int_df_p2_redd_chnge) %>%
  left_join(., int_df_p2_all_avail) %>%
  left_join(., int_df_p2_all_invol) %>%
  left_join(., int_df_p2_all_chnge) %>%
  mutate(across(all_of(names(.)), ~replace_na(., 0))) %>%
  mutate(Period = 1)


# same for second period is not possible because REDD type and Intervention codes
# for mapping to SVI are missing. Code below assumes all sampled interventions
# are REDD, but this is a stoo stroing assumption. Better data rquired.

# import p3.treat
# p3.treat <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"), sheet = "q5b_intervention_effect")
# p3.treat$Code_form <- p3.treat$P3H_code_form
# 
# # for p3, there is no data on when the intervention started. 
# # So, I will assume that it either continued from p2 or started in 2018
# # But it also does not specify whether its REDD or not, 
# # so I will assume all interventions in the sheet are REDD related.
# p3.treat$Year_implemented_earliest_in_initiative <- 2018
# 
# 
# int_df_p3 <- data.frame()
# 
# int_df_p3_avail <- int_loop_fun(p3.treat, expo_prefix = "avail", years = 2018:2020)
# 
# int_df_p3_invol <- int_loop_fun(p3.treat %>% filter(P3H_5bq3_hh_involved == 1),
#                                 expo_prefix = "invol", years = 2018:2020)
# 
# int_df_p3_chnge <- int_loop_fun(p3.treat %>% filter(P3H_5bq3a_affect_landuse == 1),
#                                 expo_prefix = "chnge", years = 2018:2020)
# 
# int_df_p3 <- expand.grid(Code_form = unique(p3.treat$P3H_code_form),
#                          year = 2018:2020) %>%
#   left_join(., int_df_p3_avail) %>%
#   left_join(., int_df_p3_invol) %>%
#   left_join(., int_df_p3_chnge) %>%
#   # replace NA by zero
#   mutate(across(all_of(names(.)), ~replace_na(., 0))) %>%
#   mutate(Period = 2)
# 
# # overwrite years 2018-2020 from p2 data with p3, matching households
# int_df_full <- int_df_p2 %>%
#   filter((!(Code_form %in% unique(int_df_p3$Code_form))) | (year < 2018)) %>%
#   bind_rows(int_df_p3)

# instead, continue with p2 data only
int_df_full <- int_df_p2


# make categorical variables aggregating interventions
exp_grad <- c("avail_", "invol_", "chnge_", "availALL", "involALL", "chngeALL")

cat_pos <- c("_NCLE", "_CLE")
cat_ena <- c("_FE", "_TC", "_OI", "_EE")
cat_neg <- c("_RFAC")
cat_tot <- c(cat_pos, cat_ena, cat_neg)

# add a zero column so that apply works (needs two columns
int_df_full$avail_invol_chnge_availALL_involALL_chngeALL_RFAC_sum <- 0

# make some indicator variables
int_cols <- names(int_df_full)

for (exp in exp_grad) {
  # Only REDD interventions
  int_df_full[[paste0("int_cat_tot_", exp)]] <- apply(int_df_full[,which(grepl(paste0(cat_tot, collapse = "|"), int_cols) & grepl(exp, int_cols) & grepl("sum", int_cols))], 1, sum)
  int_df_full[[paste0("int_cat_pos_", exp)]] <- apply(int_df_full[,which(grepl(paste0(cat_pos, collapse = "|"), int_cols) & grepl(exp, int_cols) & grepl("sum", int_cols))], 1, sum)
  int_df_full[[paste0("int_cat_ena_", exp)]] <- apply(int_df_full[,which(grepl(paste0(cat_ena, collapse = "|"), int_cols) & grepl(exp, int_cols) & grepl("sum", int_cols))], 1, sum)
  int_df_full[[paste0("int_cat_neg_", exp)]] <- apply(int_df_full[,which(grepl(paste0(cat_neg, collapse = "|"), int_cols) & grepl(exp, int_cols) & grepl("sum", int_cols))], 1, sum)
}

# remove trailing _ from variable names
names(int_df_full) <- gsub("_$", "", names(int_df_full))
int_df_full$avail_invol_chnge_availALL_involALL_chngeALL_RFAC_sum <- NULL

save(int_df_full, file = "./data/int_df_full.RData")

ggplot(int_df_full, 
       aes(x = year, 
           y = Code_form,, 
           fill = int_cat_neg_invol)) +
  geom_raster() +
  theme_minimal()

ggplot(int_df_full, 
       aes(x = year, 
           y = Code_form,, 
           fill = int_cat_neg_involALL)) +
  geom_raster() +
  theme_minimal()


##### Aggregate to village level #####
load("./data/rdat/int_df_full.RData")
load("./data/rdat/hh_pd_full.RData")


# identify variables to aggregate
trt_vars_bin <- names(int_df_full) %>%
  subset(grepl("int_", .)) %>%
  subset(grepl("bin_", .))

trt_vars_cont <- names(int_df_full) %>%
  subset(grepl("int_", .)) %>%
  subset(grepl("sum|cat", .))

# get villges from p2 data
code_villages <- hh_pd_full %>%
  select(Code_form, Village) %>%
  distinct() %>%
  filter(!duplicated(Code_form))

vil_treat_agg <- int_df_full %>%
  left_join(., code_villages) %>%
  group_by(Village, year) %>%
  summarise(n_vil = n(),
            across(all_of(trt_vars_bin), ~(mean(.x, na.rm = TRUE))*1 / n_vil),
            across(all_of(trt_vars_cont), ~(sum(.x, na.rm = TRUE))*1 / n_vil)) %>%
  ungroup()


save(vil_treat_agg, file = "./data/rdat/vil_treat_agg.RData")

# Plot treatment intensity over time

# plot treatment development over time
# ggplot(int_df_full, 
#        aes(x = year, y = Code_form, fill = factor(int_avail_bin_CLE))) +
#   # manual fill
#   scale_fill_manual(values = c("grey", "blue")) +
#   geom_raster() +
#   theme_minimal()
# 
# ggplot(int_df_full, 
#        aes(x = year, y = Code_form, fill = int_chnge_sum_CLE)) +
#   # manual fill
#   #scale_fill_manual(values = c("grey", "blue")) +
#   geom_raster() +
#   theme_minimal()

# plot treatment development over time
vil_trt_plt_pos_invol <- ggplot(vil_treat_agg, 
                                aes(x = year, 
                                    y = Village, 
                                    fill = int_cat_pos_invol)) +
  geom_raster() +
  theme_minimal()

vil_trt_plt_neg_invol <- ggplot(vil_treat_agg, 
                                aes(x = year, 
                                    y = Village, 
                                    fill = int_cat_neg_invol)) +
  geom_raster() +
  theme_minimal()

vil_trt_plt_neg_involALL <- ggplot(vil_treat_agg, 
                                aes(x = year, 
                                    y = Village, 
                                    fill = int_cat_neg_involALL)) +
  geom_raster() +
  theme_minimal()

ggsave("./out/vil_trt_plt_pos_invol.png", 
       vil_trt_plt_pos_invol, 
       height = 15, width = 8, bg = "white")


#corrplot(cor(pd[,c(int_vars, "int_count_avail","int_count_invol","int_count_chnge")]))
