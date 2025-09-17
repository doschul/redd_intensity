# run synthetic control analysis

install.packages("Synth")
install.packages("gsynth")
library(gsynth)

gsynth::gsynth()

# subset to complete cases in int_cat_ena_invol columns
vil_pd <- vil_pd[!is.na(vil_pd$int_cat_ena_invol),]

# create binary treatment variable
vil_pd$bin_ena <- ifelse(vil_pd$int_cat_pos_invol>0, 1, 0)

table(vil_pd$bin_ena, vil_pd$year)

vil_pd$bin_treat <- ifelse(vil_pd$year >= 2008 & vil_pd$Village_Ty=="Intervention", 1, 0)


out <- gsynth(jrc_perc_UndisturbedForest ~ bin_ena + tot_sum_prec + pop + jrc_fc_b_0_5000, 
              data = vil_pd,  
              index = c("vil_clean","year"), 
              se = TRUE, inference = "parametric", 
              r = c(0, 5), CV = TRUE, force = "two-way", 
              nboots = 500, seed = 42)

plot(out, type = "raw", xlab = "Year", ylab = "REDD")
plot(out, type = "gap")
