
loc_vars <- c("Code_form", "Country_code","Country_code_desc","Project_code","Project_code_desc","District","Village",
              "Village_Type","Household_code", "Latitude","Longitude", "Period")

hh_ctrls <- c(#"Distance_min", "Distance_km", 
              # HH personal characteristics
              "Marital_status","Year_formed", "Hh_born","Hh_lived",
              "Hh_ethnic","Hh_spouse" ,"Hh_spouse_ethnic", "Hh_cooking_tech",
              # HH Assets
              "Hh_floor2","Hh_walls2","Hh_roof2","Hh_own_in","Hh_own_out",
              "Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel")

outcomevars <- c("Hh_clear", "Hh_clear_total1","Hh_clear_total2","Hh_clear_total3", "perwell_comp", "income_suff")

outcomes_of_interest <- c("GFC: Def. rate (% of 2000 FC)" = "gfc_def_perc_fc2000",
                          "GFC: Forest cover (rel. to 2000)" = "gfc_remaining_forest_perc",
                          "JRC: Forest cover (rel. to 2000)" = "jrc_perc_UndisturbedForest",
                          "JRC: Total def. (% of 2000 FC)" = "jrc_perc_DefTotal",
                          "JRC: Degr. rate (% of 2000 FC)" = "jrc_perc_ForestDegradation",
                          "JRC: Def. rate (% of 2000 FC)" = "jrc_perc_DirectDeforestation",
                          "JRC: DegrDef rate (% of 2000 FC)" = "jrc_perc_DeforAfterDegrad", 
                          "SR: Deforestation rate (% of FC)" = "sf_def_perc",
                          "SR: Any clearing (%)" = "sr_def_bin_perc",
                          "SR: Forest cover (% of total area)" = "sr_forest_perc")

trtvars <- c(
  # Binary
  "int_avail_bin_CLE","int_avail_bin_EE","int_avail_bin_FE","int_avail_bin_NCLE","int_avail_bin_RFAC","int_avail_bin_TC","int_avail_bin_OI",
  "int_invol_bin_CLE","int_invol_bin_EE","int_invol_bin_FE","int_invol_bin_NCLE","int_invol_bin_RFAC","int_invol_bin_TC","int_invol_bin_OI",
  "int_chnge_bin_CLE","int_chnge_bin_EE","int_chnge_bin_FE","int_chnge_bin_NCLE","int_chnge_bin_RFAC","int_chnge_bin_TC","int_chnge_bin_OI",
  # Count (sum)           
  "int_avail_sum_CLE","int_avail_sum_EE","int_avail_sum_FE","int_avail_sum_NCLE","int_avail_sum_RFAC","int_avail_sum_TC","int_avail_sum_OI",
  "int_invol_sum_CLE","int_invol_sum_EE","int_invol_sum_FE","int_invol_sum_NCLE","int_invol_sum_RFAC","int_invol_sum_TC","int_invol_sum_OI",
  "int_chnge_sum_CLE","int_chnge_sum_EE","int_chnge_sum_FE","int_chnge_sum_NCLE","int_chnge_sum_RFAC","int_chnge_sum_TC","int_chnge_sum_OI",
  # Categories           
  "int_cat_tot_avail", "int_cat_pos_avail", "int_cat_ena_avail", "int_cat_neg_avail",
  "int_cat_tot_invol", "int_cat_pos_invol", "int_cat_ena_invol", "int_cat_neg_invol",
  "int_cat_tot_chnge", "int_cat_pos_chnge", "int_cat_ena_chnge", "int_cat_neg_chnge")

voi <- c(loc_vars, hh_ctrls, outcomevars)


