rm(list = ls())

library(tidyverse)


source("./src/redd_int_funs.R")
source("./src/varlabs.R")
#load("./data/rdat/hh_pd_bal.RData")
load("./data/rdat/hh_pd_imp.RData")
load("./data/rdat/int_df_full.RData")


# run REDD only or not
REDD_only <- FALSE

# prepare data
int_df_full$Period <- NULL

hh_pd_full <- hh_pd_imp %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018))

hh_baseline_with_forest <- hh_pd_full %>%
  filter(Period == 0,
         !is.na(forest_share),
         forest_share > 0) %>%
  pull(Code_form) %>% unique()

# restrict to baseline forest
pd <- hh_pd_full %>%
  #filter(Code_form %in% hh_baseline_with_forest) %>%
  left_join(., int_df_full, by = c("Code_form", "year")) %>% 
  filter(!is.na(int_cat_tot_availALL)) %>%
  mutate(country_project = paste0(Country_code, "_", Project_code))




# 1) calculate indices and derived variables.
# 2) impute missing values as village means.
# 3) LASSO selection on baseline forest cover outcome (P=1)
#    -> this should include ended and ongoing pre-REDD interventions
# 4) LASSO selection on treatment (number REDD interventions)
# 5) Define X and Z for later use. prepare data for wide format (crosssection)
# 5) Three datasets: Short (all), Short (BRA, PER, IND), Medium (BRA, PER, IND)


#### 1) Add new variables ####
X_vars <- c("Marital_status", "Year_formed", "Hh_born", "Hh_lived", "Hh_ethnic", 
            "Hh_own_in", "Hh_own_out", "hh_head_female", "hh_head_age","hh_member",
            "hh_dependency_ratio", "hh_head_occup_agric",
            "land_tot", "total_asset_value", "total_salary","agric_salary")



#### 2) Handle missing values ####




#### 5) Subset and export data ####

# make balanced panel short and medium term (all ids in Period 0, 1 or 0, 2
pd_short <- pd %>%
  filter(Period %in% c(0, 1)) %>%
  group_by(Code_form) %>%
  filter(n_distinct(Period) == 2) %>%
  # create time invariant treatment indicator and outcome change
  mutate(int_cat_tot_invol = int_cat_tot_avail[Period == 1],
         int_cat_tot_involALL = int_cat_tot_involALL[Period == 1],
         int_cat_tot_avail = int_cat_tot_avail[Period == 1],
         int_cat_tot_availALL = int_cat_tot_availALL[Period == 1],
         DY_forest_share = forest_share[Period == 1] - forest_share[Period == 0],
         DY_perwell_comp = perwell_comp[Period == 1],
         DY_income_suff = income_suff[Period == 1]) %>%
  ungroup() %>%
  filter(Period == 0,
         !is.na(DY_forest_share))

pd_medium <- pd %>%
  filter(Period %in% c(0, 2)) %>%
  group_by(Code_form) %>%
  filter(n_distinct(Period) == 2) %>%
  # create time invariant treatment indicator and outcome change
  mutate(int_cat_tot_invol = int_cat_tot_avail[Period == 2],
         int_cat_tot_involALL = int_cat_tot_involALL[Period == 2],
         int_cat_tot_avail = int_cat_tot_avail[Period == 2],
         int_cat_tot_availALL = int_cat_tot_availALL[Period == 2],
         DY_forest_share = forest_share[Period == 2] - forest_share[Period == 0],
         DY_perwell_comp = perwell_comp[Period == 2],
         DY_income_suff = income_suff[Period == 2]) %>%
  ungroup() %>%
  filter(Period == 0,
         !is.na(DY_forest_share))


# save for analysis in python
write_csv(pd_short, "./data/pd_short_ALL.csv")
write_csv(pd_medium, "./data/pd_medium_ALL.csv")

# identify missing values per variable
missing_vars <- pd_short %>%
  select(-c(Country_code, Project_code, Code_form, Period, year)) %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  filter(missing_count > 0)



# reload data
pd_short <- read_csv("./data/pd_short_ALL.csv")
pd_medium <- read_csv("./data/pd_medium_ALL.csv")


table(pd_short$Village_Type, pd_short$int_cat_tot_invol)
table(pd_medium$Village_Type, pd_medium$int_cat_tot_invol)


# histogram of int_cat_tot_invol by Vilage_Type
ggplot(pd_short[pd_short$Village_Type==1,], 
       aes(x = int_cat_tot_invol, fill = factor(Project_code_desc))) +
  geom_density() +
  labs(title = "Distribution of REDD interventions by Village Type",
       x = "Number of REDD interventions",
       y = "Count") +
  theme_minimal() +
  # facet bz countrz
  facet_wrap(~Country_code_desc)


# run LASSO to idrntify predictors of REDD
library(glmnet)
# function to run lasso and return selected variables

run_lasso <- function(df, outcome, covariates, alpha = 1, family_type = "gaussian") {
  
  # Create a matrix of predictors. `model.matrix` includes an intercept,
  # which `glmnet` handles correctly.
  X <- model.matrix(as.formula(paste("~", paste0(covariates, collapse = "+"), sep = "")), data = df)
  
  # Create a vector of outcomes
  y <- df[[outcome]]
  
  # Run LASSO with cross-validation to find the optimal `lambda`.
  # `cv.glmnet` automatically selects the best lambda based on the specified family.
  cat(paste0("Running cross-validated LASSO with `family = \"", family_type, "\"`...\n"))
  lasso_model <- cv.glmnet(X, y, alpha = alpha, family = family_type)
  
  # Plot the cross-validation results to visualize the error vs. log(lambda).
  par(mfrow = c(2, 1))
  plot(lasso_model, sub = "Cross-Validation Plot")
  
  # Add a second plot showing the coefficient paths as lambda decreases.
  # This visualizes how each predictor's coefficient shrinks towards zero.
  plot(lasso_model$glmnet.fit, xvar = "lambda", label = TRUE, sub = "Coefficient Paths")
  
  # Select the best predictors by finding coefficients that are not zero
  # at the optimal `lambda.min`.
  coefficients <- coef(lasso_model, s = "lambda.1se")
  selected_vars <- names(coefficients[which(coefficients != 0), ])
  
  # Remove the "(Intercept)" term from the selected variables list.
  selected_vars <- selected_vars[selected_vars != "(Intercept)"]
  
  return(selected_vars)
}

# run lasso for outcome and treatment
covariates <- c(X_vars, "forest_share", "int_nonREDD_ongoing_sum", "int_nonREDD_ended_sum")
outcome <- "DY_forest_share"
treatment <- "int_cat_tot_invol"

# run lasso for outcome
run_lasso(df = pd_short, outcome = outcome, covariates = covariates)
run_lasso(df = pd_short, outcome = "Village_Type", covariates = covariates, family_type = "binomial")
run_lasso(df = pd_short, outcome = treatment, covariates = covariates, family_type = "poisson")

run_lasso(df = pd_medium, outcome = outcome, covariates = covariates)
run_lasso(df = pd_medium, outcome = "Village_Type", covariates = covariates, family_type = "binomial")
run_lasso(df = pd_medium, outcome = treatment, covariates = covariates, family_type = "poisson")
