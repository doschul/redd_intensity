# Collection of functions for REDD intensity analysis


make_step_datasets = function(dat, 
                              horizon, 
                              treatvars){
  
  # identify households surveyed in baseline and first followup
  wave_1_idx <- intersect(dat$Code_form[dat$Period == 0], 
                          dat$Code_form[dat$Period == 1])
  
  wave_2_idx <- intersect(dat$Code_form[dat$Period == 0], 
                          dat$Code_form[dat$Period == 2])
  
  # create dummy variables for treatment
  step_treatvars <- NULL
  
  # loop over each treatment var
  for (i in 1:length(treatvars)) {
    dat <- fastDummies::dummy_cols(dat, select_columns = treatvars[i], remove_first_dummy = TRUE)
    
    # get names of the new dummy variables
    step_treatvars_tmp <- paste0(treatvars[i], "_", sort(unique(pull(dat, treatvars[i])))) %>%
      subset(!grepl("_0", .))
    
    # append to list
    step_treatvars <- c(step_treatvars, step_treatvars_tmp)
  }
  
  
  dat.short <- dat %>%
    filter(Code_form %in% wave_1_idx) %>%
    filter(Period < 2) %>%
    group_by(Code_form) %>%
    arrange(Period) %>%
    # Fill any NA values in step_treatvars with zero
    mutate(across(all_of(step_treatvars), ~replace_na(., 0))) %>%
    # create time invariant treatment indicator
    mutate(
      across(all_of(step_treatvars),
             ~ .x[Period == 1][1])
    ) %>%
    # calculate outcome change variables
    mutate(DY_forest_share = forest_share[Period == 1] - forest_share[Period == 0],
           #DY_agric_share = ag_share[Period == 1] - ag_share[Period == 0],
           post_clearing_any = Hh_clear[Period == 1],
           post_clearing_area = hh_clearing_total[Period == 1], 
           perwell_comp = perwell_comp[Period == 1],
           income_suff = income_suff[Period == 1]) %>%
    ungroup() %>%
    filter(Period == 0)
  
  
  if(horizon == "short"){
    return(list(data = dat.short,
                step_treatvars = step_treatvars))
  } 
  
  if(horizon == "long"){
    dat.long <- dat %>%
      filter(Code_form %in% wave_2_idx) %>%
      filter(!Period == 1) %>%
      group_by(Code_form) %>%
      arrange(Period) %>%
      # create time invariant treatment indicator
      mutate(
        across(all_of(step_treatvars),
               ~ .x[Period == 2][1])
      ) %>%
      # calculate outcome change variables
      mutate(DY_forest_share = forest_share[Period == 2] - forest_share[Period == 0],
             #DY_agric_share = ag_share[Period == 1] - ag_share[Period == 0],
             post_clearing_any = Hh_clear[Period == 2],
             post_clearing_area = hh_clearing_total[Period == 2],
             perwell_comp = perwell_comp[Period == 2],
             income_suff = income_suff[Period == 2]) %>%
      ungroup() %>%
      filter(Period == 0)
    
    return(list(data = dat.long, 
                step_treatvars = step_treatvars))
  }
}

exclude_nonparts = function(data, 
                            treat, 
                            village_indicator = "Village_Type"){
  # function to identify and remove non participants in treated villages
  
  # identify non-participants in treated villages
  non_participants <- data %>%
    filter(!!sym(village_indicator) == 1 & !!sym(treat) == 0) %>%
    pull(Code_form)
  
  # remove non-participants
  data <- data %>%
    filter(!Code_form %in% non_participants)
  
  return(data)
}

match_HH = function(data, 
                    treatvar, 
                    matchvars, 
                    discard = "none", 
                    method = "nearest", 
                    distance = "mahalanobis"){
  # function to match households to nearest neighbor in same project based on given variables
  
  # match households
  formula_matching <- as.formula(paste(treatvar, "~", paste(matchvars, collapse = "+")))
  
  matched <- matchit(formula_matching, 
                     data = data, 
                     method = method,
                     distance = distance,
                     exact = c("Project_code"),
                     replace = TRUE,
                     discard = discard)
  
  # get matched data
  matched_data <- match.data(matched)
  
  return(matched_data)
}

reg_report = function(data, 
                      treatvar, 
                      outvar, 
                      do_match = TRUE, 
                      matchvars, 
                      ctrlvars = NULL,
                      ...){
  
  cat(paste(treatvar, "\n"))
  # function to extract and report regression results
  
  # 1) remove non-participants in treated villages
  d <- exclude_nonparts(data, treatvar)
  
  # 2) match households
  if(do_match){
    
    # try matching, if result in error use original data and set do_match to FALSE
    tryCatch({
      # 2) match households
      d <- match_HH(d, treatvar, matchvars)
      
      did_matching <- TRUE
      
      #cat(paste("Matched sample:", nrow(d), "households\n"))
    }, error = function(e){
      cat("Matching failed, using original data\n")
    },
    did_matching <- FALSE)
  }
  
  # try lmer model, if error return NA
  tryCatch({
    # lmer requires more than one cluster for random effects.
    # if only one project is in data, use lm instead
    if(length(unique(d$country_project)) > 1) {
      did_formula <- as.formula(paste(outvar, "~", paste0(c(treatvar, ctrlvars, "(1|country_project)"), collapse = "+")))
      model <- lme4::lmer(did_formula, data = d)
    } else {
      did_formula <- as.formula(paste(outvar, "~", paste0(c(treatvar, ctrlvars), collapse = "+")))
      model <- lm(did_formula, data = d)
    }
    
    # Extract specific model coefficients
    tidy_tmp <- tidy(model) %>%
      as.data.frame() %>%
      filter(term %in% c(treatvar)) %>%
      mutate(matching = did_matching,
             outcome = outvar, 
             ctrl = paste(ctrlvars, collapse = ", "),
             n_treated = sum(d[[treatvar]] > 0),
             n_comparison = sum(d[[treatvar]] == 0))
    
    # save name of input data object as variable
    tidy_tmp$data <- deparse(substitute(data))
    
    return(tidy_tmp)
    
  }, error = function(e){
    cat("Model failed, returning NA\n")
    return(NA)
  })
}

# correlation analysis
qqplot_fun <- function(data, 
                       outcome_1, 
                       outcome_2, 
                       title = NULL) {
  
  lab_1 <- names(outcomes_of_interest[outcomes_of_interest==outcome_1])
  lab_2 <- names(outcomes_of_interest[outcomes_of_interest==outcome_2])
  
  max_val <- max(data[[outcome_1]], data[[outcome_2]])
  
  ggplot(data, aes(x = !!sym(outcome_1), y = !!sym(outcome_2))) +
    geom_point() +
    ggtitle(title) +
    labs(x = lab_1, y = lab_2) +
    geom_abline(intercept = 0, slope = 1, color = "black", lwd = 1.25) +
    geom_smooth(method = "lm", se = TRUE, color = "grey", alpha = 0.5) +
    #geom_text(aes(x = 0.2, y = 0.8, label = paste("r = ", round(cor(!!sym(outcome_1), !!sym(outcome_2)), 2))), color = "black", size = 4) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) + # , expand = c(0,0)
    scale_x_continuous(limits = c(0, max_val), expand = c(0,0)) + 
    scale_y_continuous(limits = c(0, max_val), expand = c(0,0)) + 
    coord_fixed() +
    theme_minimal()
}

# function to create dummy columns
make_int_dat <- function(df, 
                         var = "CLE", 
                         int_column = "intervention_type_short", 
                         expo_prefix = "avail") {
  df %>%
    group_by(Code_form) %>%
    summarise(
      # Interventions available in the village (binary)
      !!paste0("int_", expo_prefix, "_bin_", var) := ifelse(var %in% get(int_column), 1, 0),
      # Interventions available in the village (count)
      !!paste0("int_", expo_prefix, "_sum_", var) := sum(var == get(int_column))) %>%
    ungroup()
}

# function to loop over years and interventions
int_loop_fun = function(df, 
                        out_df = data.frame(),
                        years = 2000:2020, 
                        intervention_type_short, 
                        expo_prefix = "avail") {
  int_df <- data.frame(Code_form = unique(df$Code_form))
  
  for(y in years) {
    cat(y)
    
    dat <- df %>%
      filter(Year_implemented_earliest_in_initiative <= y)
    
    # create empty data frame
    int_df <- data.frame(Code_form = unique(df$Code_form))
    
    # loop over intervention types
    for (intervention in intervention_type_short) {
      # calculate interventions available in the village
      int_df_tmp <- make_int_dat(dat, 
                                 var = intervention, 
                                 expo_prefix = expo_prefix) %>%
        mutate(year = y)
      
      # merge results
      int_df <- left_join(int_df, int_df_tmp)
    }
    # bind rows 
    out_df <- bind_rows(out_df, int_df)
  }
  return(out_df)
}

# function to turn results into visualization dataframe
make_vizdf = function(obj){
  
  trt_var <- obj$args$treatment
  
  effects <- obj$results$Effects %>%
    as.data.frame() %>%
    mutate(time = 1:nrow(.))
  placebos <- obj$results$Placebos %>%
    as.data.frame() %>%
    mutate(time = (1:nrow(.))*(-1))
  
  # bind together and add point zero
  plot_df <- rbind(effects, placebos) %>%
    bind_rows(data.frame(time = 0, Estimate = 0, SE = 0)) %>%
    mutate(trt_var = trt_var)
  
  return(plot_df)
}

# make combined ggplot function
make_comb_ggplot = function(obj_list){
  
  dodge_width <- 1/length(obj_list)
  
  viz_df <- lapply(obj_list, make_vizdf) %>%
    do.call(rbind, .) %>%
    mutate(trt_var = case_when(grepl("avail", trt_var) ~ " Availabile",
                               grepl("invol", trt_var) ~ " Involved",
                               grepl("chnge", trt_var) ~ "Changed behavior", 
                               TRUE ~ trt_var))
  
  p <- ggplot(viz_df, aes(x = time, y = Estimate, color = trt_var, group = trt_var)) +
    # add horizontal line dashed
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    geom_point(position = position_dodge(width = dodge_width)) +
    geom_line(position = position_dodge(width = dodge_width)) +
    geom_errorbar(aes(ymin = Estimate - 1.96*SE, ymax = Estimate + 1.96*SE), 
                  width = 0.1, position = position_dodge(width = dodge_width)) +
    theme_minimal() +
    labs(color = "Exposure")
  
  return(p)
}

# make correlation between 
corfun = function(df, v1, v2) {
  df = df[complete.cases(df[c(v1, v2)]),]
  
  out <- cor(as.numeric(df[[v1]]), as.numeric(df[[v2]]))
  
  cat(paste("N Obs: ", nrow(df), "\n"), 
      paste("Correlation: ", out, "\n"))
}

