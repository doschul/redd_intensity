


library(tidyverse)
library(DIDmultiplegtDYN)
library(ggpubr)

rm(list = ls())

# load data
load("./data/vil_pd.RData")

source("./src/redd_int_funs.R")

# prepare list with function arguments
datasets <- list("Global" = vil_pd,
                 "Brazil" = vil_pd %>% filter(grepl("Brazil", project)))

treatments <- c("", "ALL") # REDD no suffix

outcomes <- c("jrc_perc_UndisturbedForest", 
              "jrc_perc_ForestDegradation",
              "jrc_perc_DirectDeforestation")

exposure_levels <- c("avail", "invol", "chnge")

n_specs <- length(treatments) * length(outcomes) * length(exposure_levels) * 3

arg_list <- list()

# loop to create arglist over multiple treatments, outcomes and datasets
for (dataset in 1:length(datasets)) {
  for (treatment in 1:length(treatments)) {
    for (outcome in 1:length(outcomes)) {
      for (exposure in 1:length(exposure_levels)) {

        arg_list_tmp <- list(
          # Positive interventions
          list(df = datasets[[dataset]],
            treatment = paste0("int_cat_pos_", exposure_levels[exposure], treatments[treatment]),
            outcome = outcomes[outcome],
            control = c("int_cat_ena_availALL", "int_cat_neg_availALL", 
                        "tot_mean_prec", "tot_mean_tmmx")), 
          list(df = datasets[[dataset]],
            treatment = paste0("int_cat_ena_", exposure_levels[exposure], treatments[treatment]),
            outcome = outcomes[outcome],
            control = c("int_cat_pos_availALL", "int_cat_neg_availALL", 
                        "tot_mean_prec", "tot_mean_tmmx")),
          list(df = datasets[[dataset]],
            treatment = paste0("int_cat_neg_", exposure_levels[exposure], treatments[treatment]),
            outcome = outcomes[outcome],
            control = c("int_cat_ena_availALL", "int_cat_pos_availALL", 
                        "tot_mean_prec", "tot_mean_tmmx")))

        arg_list <- c(arg_list, arg_list_tmp)
      }
    }
  }
}

# print second and third object from each list element as dataframe
arg_df <- lapply(arg_list, function(arg){
  arg[2:3] %>% as.data.frame()
}) %>% bind_rows() %>% 
  mutate(id = 1:nrow(.),
         dataset = rep(c("Global", "Brazil"), each = n_specs),
         interventions = ifelse(grepl("ALL", treatment), "ALL", "REDD"),
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

# check if confidence bound columns have same sign
res_df$ci_includes_zero <- ifelse(res_df$`LB CI` * res_df$`UB CI` > 0, T, F)



# function to make and combine plots
plot_combiner = function(dict = arg_df, d, o, i, c, y_range = c(-0.2, 0.2)){
  
  for (dat  in d) {
    print(dat)
    for (out in o) {
      print(out)
      int_list <- list()
      for (int in i) {
        cat_list <- list()
        for (cat in c) {
          p.idx <- dict %>% filter(dataset %in% dat,
                                     outcome %in% out,
                                     interventions %in% int,
                                     category %in% cat) %>% pull(id)
          
          p.tmp <- make_comb_ggplot(sdid_res[p.idx]) + 
            ylab(cat) + xlab(NULL) + coord_cartesian(ylim=y_range)
          
          cat_list <- append(cat_list, list(p.tmp))
        }
        # combine plots in catlist
        p.comb.cat <- ggarrange(plotlist = cat_list,
                                nrow = length((c)), common.legend = T, align = "hv")
        p.comb.cat <- annotate_figure(p.comb.cat, top = text_grob(paste0(int),
                                                       face = "bold", size = 12))
        
        int_list <- append(int_list, list(p.comb.cat))
        
      }

      p.comb.out <- ggarrange(plotlist = int_list,
                              ncol = 2, common.legend = T, align = "hv")
      
      return(p.comb.out)
    }
  }
}

# Global, Forest cover
p.glob.fc <- plot_combiner(d = c("Global"), 
              o = c("jrc_perc_UndisturbedForest"),
              i = c("ALL", "REDD"), 
              c = c("Positive", "Enabling", "Negative"), 
              y_range = c(-0.1, 0.1))

p.glob.fc <- annotate_figure(p.glob.fc, top = text_grob("Global, Forest cover",
                                                       face = "bold", size = 16))

p.glob.def <- plot_combiner(d = c("Global"), 
              o = c("jrc_perc_DirectDeforestation"),
              i = c("ALL", "REDD"), 
              c = c("Positive", "Enabling", "Negative"), 
              y_range = c(-0.03, 0.02))

p.glob.def <- annotate_figure(p.glob.def, top = text_grob("Global, Deforestation",
                                                        face = "bold", size = 16))

p.glob.deg <- plot_combiner(d = c("Global"), 
                            o = c("jrc_perc_ForestDegradation"),
                            i = c("ALL", "REDD"), 
                            c = c("Positive", "Enabling", "Negative"), 
                            y_range = c(-0.03, 0.02))

p.glob.deg <- annotate_figure(p.glob.deg, top = text_grob("Global, Degradation",
                                                          face = "bold", size = 16))

# save them individually
ggsave("./out/fig/sdid_global_fc.png", p.glob.fc, width = 8, height = 8, bg = "white")
ggsave("./out/fig/sdid_global_def.png", p.glob.def, width = 8, height = 8, bg = "white")
ggsave("./out/fig/sdid_global_deg.png", p.glob.deg, width = 8, height = 8, bg = "white")


# Brazil
p.bra.fc <- plot_combiner(d = c("Brazil"), 
              o = c("jrc_perc_UndisturbedForest"),
              i = c("ALL", "REDD"), 
              c = c("Positive", "Enabling"), 
              y_range = c(-0.2, 0.2))

p.bra.fc <- annotate_figure(p.bra.fc, top = text_grob("Brazil, Forest cover",
                                                      face = "bold", size = 16))

p.bra.def <- plot_combiner(d = c("Brazil"),
              o = c("jrc_perc_DirectDeforestation"),
              i = c("ALL", "REDD"), 
              c = c("Positive", "Enabling"), 
              y_range = c(-0.05, 0.05))

p.bra.def <- annotate_figure(p.bra.def, top = text_grob("Brazil, Deforestation",
                                                       face = "bold", size = 16))

p.bra.deg <- plot_combiner(d = c("Brazil"),
                           o = c("jrc_perc_ForestDegradation"),
                           i = c("ALL", "REDD"), 
                           c = c("Positive", "Enabling"), 
                           y_range = c(-0.05, 0.05))

p.bra.deg <- annotate_figure(p.bra.deg, top = text_grob("Brazil, Degradation",
                                                        face = "bold", size = 16))

# save them individually
ggsave("./out/fig/sdid_brazil_fc.png", p.bra.fc, width = 8, height = 6, bg = "white")
ggsave("./out/fig/sdid_brazil_def.png", p.bra.def, width = 8, height = 6, bg = "white")
ggsave("./out/fig/sdid_brazil_deg.png", p.bra.deg, width = 8, height = 6, bg = "white")



