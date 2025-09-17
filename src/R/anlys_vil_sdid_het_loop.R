library(tidyverse)
library(DIDmultiplegtDYN)
library(ggpubr)

rm(list = ls())

# load data
load("./data/rdat/vil_pd.RData")

source("./src/redd_int_funs.R")

# prepare list with function arguments
datasets <- list("Global" = vil_pd)

treatments <- c("") # REDD no suffix

outcomes <- c("jrc_perc_UndisturbedForest")

exposure_levels <- c("avail", "invol", "chnge")

n_specs_het <- length(treatments) * length(outcomes) * length(exposure_levels)

arg_list <- list()

# function to make low and high intensity datasets
make_intensity_data <- function(df, treatment, intensity) {
  
  trt.median <- quantile(df[[treatment]][df[[treatment]] > 0], probs = 0.5, na.rm = TRUE)
  trt.mean <- mean(df[[treatment]][df[[treatment]] > 0], na.rm = TRUE)
  trt.min <- min(df[[treatment]][df[[treatment]] > 0], na.rm = TRUE)
  trt.max <- max(df[[treatment]][df[[treatment]] > 0], na.rm = TRUE)
  
  # classify treatment intensity into zero, below median, above median
  df$treat_int <- case_when(df[[treatment]] > 0 & df[[treatment]] <= trt.median ~ 1,
                            df[[treatment]] > trt.median ~ 2,
                            TRUE ~ 0)
  
  
  # make highint by group (time invarying)
  df$treat_int_group <- ave(df$treat_int, df$Village, FUN = max)
  
  cat("Treatment median (min/max): ", trt.median, " (", trt.min, "/", trt.max, ") \n")
  cat("High intensity: ", sum(df$treat_int_group == 2), "\n")
  cat("Low intensity: ", sum(df$treat_int_group == 1), "\n")
  cat("Control: ", sum(df$treat_int_group == 0), "\n")
  
  # remove non-complying villages, i.e., assigned to control but actually should have received
  df <- df %>% filter(Village_Ty == "Control" | treat_int_group > 0)
  cat("Control (compliant): ", sum(df$treat_int_group == 0), "\n")
  
  d.low  <- df %>% filter(treat_int_group %in% c(0, 1))
  d.high <- df %>% filter(treat_int_group %in% c(0, 2))
  
  if(intensity == "low") {
    return(d.low)
  } else if(intensity == "high") {
    return(d.high)
  } else {
    stop("intensity must be 'low' or 'high'")
  }
}



# loop to create arglist over multiple treatments, outcomes and datasets
for (dataset in 1:length(datasets)) {
  for (treatment in 1:length(treatments)) {
    for (outcome in 1:length(outcomes)) {
      for (exposure in 1:length(exposure_levels)) {
        
        arg_list_tmp <- list(
            # Positive interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_pos_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "low"),
                 treatment = paste0("int_cat_pos_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_ena_availALL", "int_cat_neg_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")), 
            # Enabling interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_ena_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "low"),
                 treatment = paste0("int_cat_ena_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_pos_availALL", "int_cat_neg_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")),
            # Negative interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_neg_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "low"),
                 treatment = paste0("int_cat_neg_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_ena_availALL", "int_cat_pos_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")),
            
            # Positive interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_pos_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "high"),
                 treatment = paste0("int_cat_pos_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_ena_availALL", "int_cat_neg_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")), 
            # Enabling interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_ena_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "high"),
                 treatment = paste0("int_cat_ena_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_pos_availALL", "int_cat_neg_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")),
            # Negative interventions
            list(df = make_intensity_data(df = datasets[[dataset]],
                                          treatment = paste0("int_cat_neg_", exposure_levels[exposure], treatments[treatment]),
                                          intensity = "high"),
                 treatment = paste0("int_cat_neg_", exposure_levels[exposure], treatments[treatment]),
                 outcome = outcomes[outcome],
                 control = c("int_cat_ena_availALL", "int_cat_pos_availALL", 
                             "tot_mean_prec", "tot_mean_tmmx")))
        
        print(length(arg_list_tmp))
          
        arg_list <- c(arg_list, arg_list_tmp)
        
        print(length(arg_list))

      }
    }
  }
}

# print second and third object from each list element as dataframe
arg_df <- lapply(arg_list, function(arg){
  arg[2:3] %>% as.data.frame()
}) %>% bind_rows() %>% 
  mutate(id = 1:nrow(.),
         dataset = rep(c("Global"), each = n_specs_het * 6),
         interventions = rep(rep(c("Low intensity","High intensity"), each = 3), n_specs_het),
         exposure = case_when(
           grepl("avail", treatment) ~ "avail",
           grepl("invol", treatment) ~ "invol",
           grepl("chnge", treatment) ~ "chnge"),
         category = case_when(
           grepl("pos", treatment) ~ "Positive",
           grepl("ena", treatment) ~ "Enabling",
           grepl("neg", treatment) ~ "Negative"))

arg_df


sdid_res <- lapply(arg_list, function(arg){
  
  print(arg[2:3])
  
  # try function, if error return NULL
  tryCatch({
    did_multiplegt_dyn(df = arg[["df"]],
                       treatment = arg[["treatment"]],
                       outcome = arg[["outcome"]],
                       control = arg[["control"]],
                       group = "Village",
                       time = "year", 
                       cluster = "project",
                       only_never_switchers = F,
                       placebo = 6, effects = 8, 
                       graph_off = T)
  }, error = function(e) {
    return(NULL)
  })
})


p.glob.fc.int <- plot_combiner(dict = arg_df,
                               d = c("Global"), 
                           o = c("jrc_perc_UndisturbedForest"),
                           i = c("Low intensity", "High intensity"), 
                           c = c("Positive", "Enabling", "Negative"), 
                           y_range = c(-0.2, 0.2))


# save results
ggsave(filename = "./out/fig/p.glob.fc.int.png", plot = p.glob.fc.int, 
       width = 10, height = 10, bg = "white")


# extract model results, if not NULL
res_df_tmp <- lapply(sdid_res, function(res){
  
  if (!is.null(res)) {
    res_df_tmp <- res$results$ATE %>%
      as.data.frame() %>%
      mutate(p_jointeffects = res$results$p_jointeffects,
             p_jointplacebo = res$results$p_jointplacebo)
    
    return(res_df_tmp)
  } else {
    return(data.frame(p_jointeffects = NA))
  }
}) %>% do.call(bind_rows, .)

# merge with arg_df
res_df <- bind_cols(arg_df, res_df_tmp)
