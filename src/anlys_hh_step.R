# Household level DID analysis using step function


# run step function dynamically (on different outcomes)

# prepare short and long term datasets
# exclude non-participants in treated villages (spillover)
# optional: matching HH
# run regressions
# plot results

rm(list = ls())

library(MatchIt)
library(lme4)
library(broom.mixed)
library(ggh4x)
library(scales)
library(metafor)
library(texreg)
library(tidyverse)


source("./src/redd_int_funs.R")
source("./src/varlabs.R")
load("./data/rdat/hh_pd_bal.RData")
load("./data/rdat/int_df_full.RData")


# run REDD only or not
REDD_only <- TRUE



# prepare data
int_df_full$Period <- NULL

hh_pd_bal <- hh_pd_bal %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018))

hh_baseline_with_forest <- hh_pd_bal %>%
  filter(Period == 0,
         !is.na(forest_share),
         forest_share > 0) %>%
  pull(Code_form) %>% unique()

# restrict to baseline forest
pd <- hh_pd_bal %>%
  #filter(Code_form %in% hh_baseline_with_forest) %>%
  left_join(., int_df_full, by = c("Code_form", "year")) %>%
  filter(!is.na(int_cat_tot_avail))


# run on all interventions (optioal)
if (!REDD_only) {
  trtvars[grepl("int_cat_", trtvars)] <- paste0(trtvars[grepl("int_cat_", trtvars)], "ALL")
}



# Step 1: Prepare short and long term datasets
dat.short <- make_step_datasets(dat = pd, horizon = "short", treatvars =  trtvars)[[1]]
dat.long <- make_step_datasets(pd, "long", trtvars)[[1]]

dat.short <- dat.short %>%
  mutate(country_project = paste(Country_code, Project_code, sep = "_"))
dat.long <- dat.long %>%
  mutate(country_project = paste(Country_code, Project_code, sep = "_"))

step_treatvars <- make_step_datasets(dat = pd, horizon = "short", treatvars =  trtvars)[[2]]
step_treatvars <- step_treatvars %>% subset(!grepl("REDD", .))



# define matching variables
matchvars <- c("forest_share", "land_tot", "hh_member", "hh_head_age", 
               "hh_dependency_ratio","hh_head_occup_agric", "hh_head_female",
               "Hh_floor2","Hh_walls2","Hh_roof2",
               "Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel"
               )

apply(dat.long[matchvars], 2, function(x) sum(is.na(x)))

# fill missing values with village mean
dat.short <- dat.short %>%
  group_by(Village) %>%
  mutate_at(vars(all_of(matchvars)), ~replace_na(., mean(., na.rm = TRUE)))

dat.long <- dat.long %>%
  group_by(Village) %>%
  mutate_at(vars(all_of(matchvars)), ~replace_na(., mean(., na.rm = TRUE)))


# identify treatment variables with at least five treated units
applicable_treatvars_short <- step_treatvars[sapply(step_treatvars, function(x) sum(dat.short[[x]] > 0) >= 5)]
applicable_treatvars_long <- step_treatvars[sapply(step_treatvars, function(x) sum(dat.long[[x]] > 0) >= 5)]


# loop over all treatment variables
res_df <- data.frame(term=NULL,
                     estimate=NULL, 
                     std.error=NULL, 
                     statistic=NULL, 
                     p.value=NULL,
                     matching = NULL,
                     outcome = NULL,
                     n_treated = NULL)

for (i in 1:length(applicable_treatvars_short)) {
  
  res_tmp <- reg_report(dat.short, 
                        treatvar = applicable_treatvars_short[i], 
                        outvar = "DY_forest_share", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}

for (i in 1:length(applicable_treatvars_long)) {
  
  res_tmp <- reg_report(dat.long, 
                        treatvar = applicable_treatvars_long[i], 
                        outvar = "DY_forest_share", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}

# make same for clearing outcome
for (i in 1:length(applicable_treatvars_short)) {
  
  res_tmp <- reg_report(dat.short, 
                        treatvar = applicable_treatvars_short[i], 
                        outvar = "post_clearing_any", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}

for (i in 1:length(applicable_treatvars_long)) {
  
  res_tmp <- reg_report(dat.long, 
                        treatvar = applicable_treatvars_long[i], 
                        outvar = "post_clearing_any", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}

# Clearing area as outcome
dat.short$post_clearing_area_log <- log(dat.short$post_clearing_area + 1)
dat.long$post_clearing_area_log <- log(dat.long$post_clearing_area + 1)


for (i in 1:length(applicable_treatvars_short)) {
  
  res_tmp <- reg_report(dat.short, 
                        treatvar = applicable_treatvars_short[i], 
                        outvar = "post_clearing_area_log", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}

for (i in 1:length(applicable_treatvars_long)) {
  
  res_tmp <- reg_report(dat.long, 
                        treatvar = applicable_treatvars_long[i], 
                        outvar = "post_clearing_area_log", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  res_df <- rbind(res_df, res_tmp)
}




res_out1 <- res_df %>%
  mutate(category = case_when(grepl("bin", term) ~ "Binary (any)",
                              grepl("sum", term) ~ "Count (sum)",
                              grepl("cat", term) ~ "Count (sum)"),
         indicator = case_when(grepl("_CLE", term) ~ " CLE",
                               grepl("EE", term) ~ " EE",
                               grepl("FE", term) ~ " FE",
                               grepl("NCLE", term) ~ " NCLE",
                               grepl("RFAC", term) ~ " RFAC",
                               grepl("TC", term) ~ " TC",
                               grepl("OI", term) ~ " OI",
                               grepl("pos", term) ~ "Positive",
                               grepl("neg", term) ~ "Negative",
                               grepl("ena", term) ~ "Enabling",
                               grepl("tot", term) ~ "Total"),
         exposure = case_when(grepl("avail", term) ~ "Available",
                           grepl("invol", term) ~ "Involved",
                           grepl("chnge", term) ~ "Changed behavior"),
         # extract number as step, set NA if no number is found
         step = as.integer(substr(term, regexpr("[0-9]", term), nchar(term)))
  ) %>%
  mutate(exposure = factor(exposure, levels = c("Available","Involved","Changed behavior")), 
         data = case_when(grepl("long", data) ~ "Long-term",
                          grepl("short", data) ~ "Short-term")) %>%
  filter(!is.na(exposure))

# save result as csv file
if (REDD_only) {
  write.csv(res_out1, file = "./out/tbl/hh_step_res_REDD.csv", row.names = FALSE)
} else if (!REDD_only) {
  write.csv(res_out1, file = "./out/tbl/hh_step_res_ALLINT.csv", row.names = FALSE)
}



# read results
REDD_only <- FALSE
res_out1 <- read.csv("./out/tbl/hh_step_res_REDD.csv") %>%
  mutate(exposure = factor(exposure, levels = c("Available","Involved","Changed behavior")))
res_out1 <- read.csv("./out/tbl/hh_step_res_ALLINT.csv") %>%
  mutate(exposure = factor(exposure, levels = c("Available","Involved","Changed behavior")))


# visualize estimates coefficients
p.step.fc <- ggplot(res_out1 %>% 
                      mutate(Significance = ifelse(abs(statistic) > 1.96, "Yes (95%)", "n.s.")) %>%
         filter(grepl("cat", term),
                outcome == "DY_forest_share"), 
       aes(x = step, y = estimate)) +
  theme_bw() + 
  geom_hline(yintercept=0, linetype="dashed") +
  #geom_point(aes(color = horizon)) +
  geom_pointrange(aes(ymin = estimate - 1.96 * std.error, 
                      ymax = estimate + 1.96 * std.error,
                      shape = Significance,
                      color = data),
                  position = position_dodge(0.3)) +
  labs(x = NULL, y = NULL, title = 'Change in forest cover DiD-estimates') +
  facet_grid(rows = vars(indicator), 
             cols = vars(exposure), 
             switch = "y", 
             scales = "fixed") +
  scale_x_continuous(breaks = breaks_pretty()) +
  coord_cartesian(ylim=c(-0.2, 0.4)) +
  theme(legend.position = "bottom")


p.step.clrb <- ggplot(res_out1 %>% 
                        mutate(Significance = ifelse(abs(statistic) > 1.96, "Yes (95%)", "n.s.")) %>%
         filter(grepl("cat", term),
                outcome == "post_clearing_any"), 
       aes(x = step, y = estimate)) +
  theme_bw() + 
  geom_hline(yintercept=0, linetype="dashed") +
  #geom_point(aes(color = horizon)) +
  geom_pointrange(aes(ymin = estimate - 1.96 * std.error, 
                      ymax = estimate + 1.96 * std.error,
                      shape = Significance,
                      color = data),
                  position = position_dodge(0.3)) +
  labs(x = NULL, y = NULL, title = 'Change in clearing DiD-estimates') +
  facet_grid(rows = vars(indicator), 
             cols = vars(exposure), 
             switch = "y", 
             scales = "fixed") +
  scale_x_continuous(breaks = breaks_pretty()) +
  coord_cartesian(ylim=c(-0.2, 0.4)) +
  theme(legend.position = "bottom")

# clearing area plot
p.step.clrarea <- ggplot(res_out1 %>% 
                        mutate(Significance = ifelse(abs(statistic) > 1.96, "Yes (95%)", "n.s.")) %>%
                        filter(grepl("cat", term),
                               outcome == "post_clearing_area_log"), 
                      aes(x = step, y = estimate)) +
  theme_bw() + 
  geom_hline(yintercept=0, linetype="dashed") +
  #geom_point(aes(color = horizon)) +
  geom_pointrange(aes(ymin = estimate - 1.96 * std.error, 
                      ymax = estimate + 1.96 * std.error,
                      shape = Significance,
                      color = data),
                  position = position_dodge(0.3)) +
  labs(x = NULL, y = NULL, title = 'Change in clearing area DiD-estimates') +
  facet_grid(rows = vars(indicator), 
             cols = vars(exposure), 
             switch = "y", 
             scales = "fixed") +
  scale_x_continuous(breaks = breaks_pretty()) +
  coord_cartesian(ylim=c(-0.5, 0.6)) +
  theme(legend.position = "bottom")

p.step.clrarea

# save plots
if (REDD_only) {
  ggsave(plot = p.step.fc, filename = "./out/fig/p.step.fc.png", 
         width = 6, height = 8)
  
  ggsave(plot = p.step.clrb, filename = "./out/fig/p.step.clrb.png", 
         width = 6, height = 8)
} else if (!REDD_only) {
  ggsave(plot = p.step.fc, filename = "./out/fig/p.step.fc_ALL.png", 
         width = 6, height = 8)
  
  ggsave(plot = p.step.clrb, filename = "./out/fig/p.step.clrb_ALL.png", 
         width = 6, height = 8)
}


##### Meta analysis #####

meta_dat_full <- res_out1 %>%
  mutate(step_sqrd = step^2,
         vi = (std.error^2), 
         id = 1:nrow(.), 
         facet_id = as.factor(paste0(indicator, exposure, data)))

# 1) change in Forest cover
meta_dat <- res_out1 %>%
  filter(category == "Count (sum)") %>%
  mutate(step_sqrd = step^2,
         vi = (std.error^2), 
         id = 1:nrow(.), 
         facet_id = as.factor(paste0(indicator, exposure, data)))

# run moderation models for exposure, indicator, and data
m.fc.avg <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id,
                            data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                            method = "REML")

m.fc.avg.bin <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id,
                            data = meta_dat_full %>% filter(outcome == "DY_forest_share", grepl("bin", term)),
                            method = "REML")

m.fc.cat.bin <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id,
                                data = meta_dat_full %>% filter(outcome == "DY_forest_share", grepl("bin", term)),
                                mods = ~ 0 + indicator,
                                method = "REML")

m.fc.exp.bin <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id,
                                data = meta_dat_full %>% filter(outcome == "DY_forest_share", grepl("bin", term)),
                                mods = ~ 0 + exposure,
                                method = "REML")

m.fc.dat <- metafor::rma.mv(yi = estimate, 
                V = vi, 
                random = ~ 1 | id / facet_id,
                data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                mods = ~ 0 + data,
                method = "REML")

m.fc.ind_cat <- metafor::rma.mv(yi = estimate, 
                      V = vi, 
                      random = ~ 1 | id / facet_id,
                      data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                      mods = ~ 0 + indicator,
                      method = "REML")

m.fc.ind_type <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "DY_forest_share", !grepl("cat", term)),
                            mods = ~ 0 + indicator,
                            method = "REML")

m.fc.exp <- metafor::rma.mv(yi = estimate, 
                      V = vi, 
                      random = ~ 1 | id / facet_id,
                      data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                      mods = ~ 0 + exposure,
                      method = "REML")

m.fc.step <- metafor::rma.mv(yi = estimate, 
                      V = vi, 
                      random = ~ 1 | id / facet_id, 
                      data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                      mods = ~ step + step_sqrd,
                      method = "REML")

m.fc.full.cat <- metafor::rma.mv(yi = estimate, 
                      V = vi, 
                      random = ~ 1 | id / facet_id,
                      data = meta_dat %>% filter(outcome == "DY_forest_share", grepl("cat", term)),
                      mods = ~ step + step_sqrd + exposure + indicator + data,
                      method = "REML")

m.fc.full.type <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id,
                             data = meta_dat %>% filter(outcome == "DY_forest_share", !grepl("cat", term)),
                             mods = ~ step + step_sqrd + exposure + indicator + data,
                             method = "REML")


coef_names <- c("Intercept", "Step","Step2", "Long-term", "Short-term", 
                "Enabling", "Negative", "Positive", "Total",
                "CLE", "EE", "FE", "NCLE", "OI", "RFAC", "TC",
                "Available", "Involved", "Changed behavior")
coef_groups <- list("Time-horizon" = 4:5, "Category" = 6:9, "Type" = 10:16, "Exposure" = 17:19)


screenreg(list(m.fc.step, m.fc.dat, m.fc.ind_cat, m.fc.ind_type, m.fc.exp, m.fc.full.cat, m.fc.full.type), 
          groups = coef_groups, 
          custom.coef.names = coef_names)




# 2) self-reported clearing

# run moderation models for exposure, indicator, and data
m.clr.dat <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "post_clearing_any", grepl("cat", term)),
                            mods = ~ 0 + data,
                            method = "REML")

m.clr.ind.cat <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "post_clearing_any", grepl("cat", term)),
                            mods = ~ 0 + indicator,
                            method = "REML")

m.clr.ind.typ <- metafor::rma.mv(yi = estimate, 
                                 V = vi, 
                                 random = ~ 1 | id / facet_id,
                                 data = meta_dat %>% filter(outcome == "post_clearing_any", !grepl("cat", term)),
                                 mods = ~ 0 + indicator,
                                 method = "REML")

m.clr.exp <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "post_clearing_any", grepl("cat", term)),
                            mods = ~ 0 + exposure,
                            method = "REML")

m.clr.step <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id,
                             data = meta_dat %>% filter(outcome == "post_clearing_any", grepl("cat", term)),
                             mods = ~ step + step_sqrd,
                             method = "REML")

m.clr.full.cat <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id,
                             data = meta_dat %>% filter(outcome == "post_clearing_any", grepl("cat", term)),
                             mods = ~ step + exposure + indicator + data,
                             method = "ML")

m.clr.full.typ <- metafor::rma.mv(yi = estimate, 
                              V = vi, 
                              random = ~ 1 | id / facet_id,
                              data = meta_dat %>% filter(outcome == "post_clearing_any", !grepl("cat", term)),
                              mods = ~ step + step_sqrd + exposure + indicator + data,
                              method = "REML")

screenreg(list(m.clr.step, m.clr.dat, m.clr.ind.cat,m.clr.ind.typ, m.clr.exp, m.clr.full.cat, m.clr.full.typ), 
          groups = coef_groups, 
          custom.coef.names = coef_names)



# Area reg

m.clrarea.step <- metafor::rma.mv(yi = estimate, 
                              V = vi, 
                              random = ~ 1 | id / facet_id,
                              data = meta_dat %>% filter(outcome == "post_clearing_area_log", grepl("cat", term)),
                              mods = ~ step + step_sqrd,
                              method = "REML")

m.clrarea.full.cat <- metafor::rma.mv(yi = estimate, 
                                  V = vi, 
                                  random = ~ 1 | id / facet_id,
                                  data = meta_dat %>% filter(outcome == "post_clearing_area_log", grepl("cat", term)),
                                  mods = ~ step + exposure + indicator + data,
                                  method = "ML")

m.clrarea.full.typ <- metafor::rma.mv(yi = estimate, 
                                  V = vi, 
                                  random = ~ 1 | id / facet_id,
                                  data = meta_dat %>% filter(outcome == "post_clearing_area_log", !grepl("cat", term)),
                                  mods = ~ step + step_sqrd + exposure + indicator + data,
                                  method = "REML")

screenreg(list(m.clrarea.step, m.clrarea.full.cat, m.clrarea.full.typ))
# save tables

if (REDD_only) {
  htmlreg(list(m.fc.step, m.fc.dat, m.fc.ind_cat, m.fc.ind_type, m.fc.exp, m.fc.full.cat, m.fc.full.type), 
          groups = coef_groups, custom.coef.names = coef_names, 
          file = "./out/tbl/hh.meta.fc.html")
  
  htmlreg(list(m.clr.step, m.clr.dat, m.clr.ind.cat,m.clr.ind.typ, m.clr.exp, m.clr.full.cat, m.clr.full.typ), 
          groups = coef_groups, custom.coef.names = coef_names, 
          file = "./out/tbl/hh.meta.clr.html")
} else if (!REDD_only) {
  htmlreg(list(m.fc.step, m.fc.dat, m.fc.ind_cat, m.fc.ind_type, m.fc.exp, m.fc.full.cat, m.fc.full.type), 
          groups = coef_groups, custom.coef.names = coef_names, 
          file = "./out/tbl/hh.meta.fc.ALL.html")
  
  htmlreg(list(m.clr.step, m.clr.dat, m.clr.ind.cat,m.clr.ind.typ, m.clr.exp, m.clr.full.cat, m.clr.full.typ), 
          groups = coef_groups, custom.coef.names = coef_names, 
          file = "./out/tbl/hh.meta.clr.ALL.html")
}



