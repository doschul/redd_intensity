# HH step and meta regression - BRAZIL ONLY

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
REDD_only <- F

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

# OPTIONAL: restrict to BRAZIL
pd <- pd %>%
  filter(Country_code == "101") # Brazil


# run on all interventions (optioal)
if (!REDD_only) {
  trtvars[grepl("int_cat_", trtvars)] <- paste0(trtvars[grepl("int_cat_", trtvars)], "ALL")
} else {
  trtvars <- trtvars[grepl("int_cat_", trtvars)]
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

# save as csv
# save data as csv
if (REDD_only) {
  write.csv(dat.short, file = "./data/rdat/hh_step_short_BRA_REDD.csv", row.names = FALSE)
  write.csv(dat.long, file = "./data/rdat/hh_step_long_BRA_REDD.csv", row.names = FALSE)
} else if (!REDD_only) {
  write.csv(dat.short, file = "./data/rdat/hh_step_short_BRA_ALLINT.csv", row.names = FALSE)
  write.csv(dat.long, file = "./data/rdat/hh_step_long_BRA_ALLINT.csv", row.names = FALSE)
}
#### 1) Describe complemenatrity REDD/non-REDD interventions ####

pd|>
  filter(year == 2018) |>
  gtsummary::tbl_summary( 
  by = Village_Type, 
  include = c("int_nonREDD_ended_sum",
              "int_nonREDD_ongoing_sum", 
              "int_cat_tot_avail"), 
  label = list(int_nonREDD_ended_sum = "Non-REDD ended",
               int_nonREDD_ongoing_sum = "Non-REDD ongoing",
               int_cat_tot_avail = "REDD available"))

# 77% of treated have no non-REDD interventions that ended, 23% have at least one 
# ongoing non-REDD intervention that ended.
# whereas 81% of controls were part of some 
# non-REDD intervention that ended.

# 68% of controls are not part of any ongoing non-REDD intervention, 
# while 100% of treated are part of at least one ongoing non-REDD.

# correlate REDD and non-REDD interventions
pd|>
  filter(year == 2018,
         Village_Type == 1) |>
  ggplot(aes(x = int_nonREDD_ended_sum, y = int_cat_tot_avail)) +
  geom_point() +
  geom_smooth(method = "lm", se = T) +
  labs(x = "Non-REDD ongoing interventions", y = "REDD involved interventions") +
  theme_bw()

pd|>
  filter(year == 2018,
         Village_Type == 1) |>
  ggplot(aes(x = int_nonREDD_ongoing_sum, y = int_cat_tot_invol)) +
  geom_point() +
  geom_smooth(method = "lm", se = T) +
  labs(x = "Non-REDD ongoing interventions", y = "REDD involved interventions") +
  theme_bw()

# density plot of ended and ongoing as facets, village type as fill
df_long <- pd|>
  filter(year == 2018) |>
  # arrange codeforms by Non-REDD ongoing
  mutate(Code_form = factor(Code_form, 
                            levels = unique(Code_form[order(int_nonREDD_ongoing_sum)]))) |>
  pivot_longer(cols = c(int_nonREDD_ended_sum, int_nonREDD_ongoing_sum,
                        int_cat_ena_avail, int_cat_neg_avail, int_cat_pos_avail), 
               names_to = "intervention_type", 
               values_to = "value") |>
  mutate(treat = ifelse(Village_Type == 1, "Treated", "Control"),
         intervention_type = case_when(
           intervention_type == "int_nonREDD_ended_sum" ~ "Non-REDD ended",
           intervention_type == "int_nonREDD_ongoing_sum" ~ "Non-REDD ongoing",
           intervention_type == "int_cat_ena_avail" ~ "Enabling",
           intervention_type == "int_cat_neg_avail" ~ "Negative",
           intervention_type == "int_cat_pos_avail" ~ "Positive"
         )) 

p.trt.int1 <- df_long |>
  filter(grepl("Non-REDD", intervention_type)) |>
  ggplot(aes(x = value, fill = treat)) +
  geom_density(alpha = 0.5) +
  # invert axes
  coord_flip(expand = FALSE) +
  # invert x axis
  scale_x_reverse() +
  facet_wrap(~ intervention_type) +
  labs(x = "Intervention count", y = "Density") +
  theme_bw() +
  labs(fill = "") +
  # legend bottom
  theme(legend.position = "bottom") +
  scale_fill_manual(values = c("Treated" = "#1f77b4", "Control" = "#ff7f0e"))

p.trt.int2 <- df_long |>
  filter(!grepl("Non-REDD", intervention_type), 
         Village_Type == 1, 
         value > 0) |>
  mutate(facet_lab = "REDD+ interventions") |>
  ggplot(aes(x = Code_form, y = value, fill = intervention_type)) +
  geom_col(position = "stack") +
  # make single facet called REDD interventions
  facet_wrap(~ facet_lab, ncol = 1) +
  # rotate plot
  coord_flip() +
  # remove x axis labels
  scale_x_discrete(labels = NULL) +
  labs(x = NULL, y = "Intervention count") +
  # legend title
  labs(fill = "") +
  #legend bottom
  theme_bw() +
  theme(legend.position = "bottom") +
  scale_fill_manual(values = c("Enabling" = "#2ca02c", 
                                "Negative" = "#d62728", 
                                "Positive" = "#9467bd"))

library(ggpubr)
p.trt.int <- ggarrange(p.trt.int1, p.trt.int2, 
                     nrow = 1, 
                     widths = c(1, 1)) +
  theme(legend.position = "bottom")

p.trt.int

  
  
  
  
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
  
  tryCatch({
    res_df <- bind_rows(res_df, res_tmp)
  }, error = function(e) {
    message(paste(e$message))
  })
}

for (i in 1:length(applicable_treatvars_long)) {
  
  res_tmp <- reg_report(dat.long, 
                        treatvar = applicable_treatvars_long[i], 
                        outvar = "DY_forest_share", 
                        do_match = TRUE, 
                        matchvars = matchvars)
  
  tryCatch({
    res_df <- bind_rows(res_df, res_tmp)
  }, error = function(e) {
    message(paste(e$message))
  })
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

p.step.fc


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
                             random = ~ 1 | id / facet_id, 
                             data = meta_dat %>% 
                               filter(outcome == "DY_forest_share", 
                                      grepl("cat", term)),
                             method = "REML")

m.fc.step <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id, 
                             data = meta_dat %>% 
                               filter(outcome == "DY_forest_share", 
                                      grepl("cat", term)),
                             mods = ~ step,
                             method = "REML")

m.fc.step2 <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id, 
                             data = meta_dat %>% 
                               filter(outcome == "DY_forest_share", 
                                      grepl("cat", term)),
                             mods = ~ step + step_sqrd,
                             method = "REML")

m.fc.full.cat <- metafor::rma.mv(yi = estimate, 
                                 V = vi, 
                                 random = ~ 1 | id / facet_id,
                                 data = meta_dat %>% 
                                   filter(outcome == "DY_forest_share", 
                                          grepl("cat", term)),
                                 mods = ~ step + step_sqrd + exposure + data,
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


screenreg(list(m.fc.avg, m.fc.step2, m.fc.full.cat), 
          #groups = coef_groups, 
          #custom.coef.names = coef_names
          )

# predict first model with new data, steo ranging from 1:15

# count max intervention including nonREDD
max(dat.short$int_nonREDD_ongoing_sum)
max(dat.short$int_cat_tot_availALL)
max(dat.short$int_cat_tot_involALL)

pred_data <- data.frame(step = 1:10, 
                         step_sqrd = (1:10)^2) |>
  as.matrix()


# plot predicted values
pred_fc <- predict(m.fc.step2, newmods = pred_data) |>
  as.data.frame() |>
  mutate(step = pred_data[,1])

ggplot(pred_fc, 
       xlab = "Step", 
       ylab = "Predicted change in forest cover",
       main = "Predicted change in forest cover over time",
       ylim = c(-0.2, 0.4)) +
  geom_hline(yintercept=0, linetype="dashed") +
  geom_line(aes(x = step, y = pred), color = "blue") +
  geom_ribbon(aes(x = step, ymin = ci.lb, ymax = ci.ub), alpha = 0.2)


# Perceived wellbeing
m.pw.avg <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id,
                            data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                            method = "REML")

m.pw.avg.bin <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id,
                                data = meta_dat_full %>% filter(outcome == "perwell_comp", grepl("bin", term)),
                                method = "REML")

m.pw.cat.bin <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id,
                                data = meta_dat_full %>% filter(outcome == "perwell_comp", grepl("bin", term)),
                                mods = ~ 0 + indicator,
                                method = "REML")

m.pw.exp.bin <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id,
                                data = meta_dat_full %>% filter(outcome == "perwell_comp", grepl("bin", term)),
                                mods = ~ 0 + exposure,
                                method = "REML")

m.pw.dat <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                            mods = ~ 0 + data,
                            method = "REML")

m.pw.ind_cat <- metafor::rma.mv(yi = estimate, 
                                V = vi, 
                                random = ~ 1 | id / facet_id,
                                data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                                mods = ~ 0 + indicator,
                                method = "REML")

m.pw.ind_type <- metafor::rma.mv(yi = estimate, 
                                 V = vi, 
                                 random = ~ 1 | id / facet_id,
                                 data = meta_dat %>% filter(outcome == "perwell_comp", !grepl("cat", term)),
                                 mods = ~ 0 + indicator,
                                 method = "REML")

m.pw.exp <- metafor::rma.mv(yi = estimate, 
                            V = vi, 
                            random = ~ 1 | id / facet_id,
                            data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                            mods = ~ 0 + exposure,
                            method = "REML")

m.pw.step <- metafor::rma.mv(yi = estimate, 
                             V = vi, 
                             random = ~ 1 | id / facet_id, 
                             data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                             mods = ~ step + step_sqrd,
                             method = "REML")

m.pw.full.cat <- metafor::rma.mv(yi = estimate, 
                                 V = vi, 
                                 random = ~ 1 | id / facet_id,
                                 data = meta_dat %>% filter(outcome == "perwell_comp", grepl("cat", term)),
                                 mods = ~ step + step_sqrd + exposure + indicator + data,
                                 method = "REML")

m.pw.full.type <- metafor::rma.mv(yi = estimate, 
                                  V = vi, 
                                  random = ~ 1 | id / facet_id,
                                  data = meta_dat %>% filter(outcome == "perwell_comp", !grepl("cat", term)),
                                  mods = ~ step + step_sqrd + exposure + indicator + data,
                                  method = "REML")

screenreg(list(m.pw.step, m.pw.dat, m.pw.ind_cat, m.pw.ind_type, m.pw.exp, m.pw.full.cat, m.pw.full.type), 
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
                                  mods = ~ step + step_sqrd + exposure + indicator + data,
                                  method = "REML", verbose = T)

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



