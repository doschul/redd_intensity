# variable import

rm(list=ls())

# libraries
library(tidyverse)
library(readxl)

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/cifor")
source("./src/varlabs.R")


datapath <- "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/1. Research/1. M2 Main Database/1. Global Database"

##### Prepare panel and ID vars #####

p1 <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), sheet = 1)
p2 <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), sheet = 1)
p3 <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"), sheet = 1)


# fit some column types
p1$Year_formed <- as.numeric(p1$Year_formed)
p1$Hh_lived <- as.numeric(p1$Hh_lived)
p1$Forest_cash <- as.numeric(p1$Forest_cash)
p1$Hh_clear_total1 <- as.numeric(p1$Hh_clear_total1)
p1$Hh_clear_total2 <- as.numeric(p1$Hh_clear_total2)
p1$Hh_clear_total3 <- as.numeric(p1$Hh_clear_total3)


# rename variables to match
p3 <- p3 %>%
  rename(Code_form = "P3H_code_form",
         Country_code = "P3H_country_code",
         Project_code = "P3H_project_code", 
         District = "P3H_district",
         Village = "P3H_village",
         Village_Type = "P3H_village_type",
         Household_code = "P3H_household_code",
         Latitude = "P3H_location_latitude",                
         Longitude ="P3H_location_longitude",
         Period = "P3H_phase",
         # hh_ctrls
         Marital_status ="P3H_1cq1_marital_status"  ,
         Year_formed  ="P3H_1cq2_hh_formed",
         Hh_born  ="P3H_1cq3_hhborn",
         Hh_lived  ="P3H_1cq4_hhlived",
         Hh_ethnic  ="P3H_1cq5_hhethnic",
         Hh_spouse  ="P3H_1cq6_spouseborn",
         Hh_spouse_ethnic  ="P3H_1cq8_spousethnic"  ,
         Hh_floor2  ="P3H_2cq2_floorval" ,
         Hh_walls2  ="P3H_2cq4_wallsval",
         Hh_roof2  ="P3H_2cq2_roofval",
         Hh_water  ="P3H_2dq1_water",
         Hh_toilet  ="P3H_2dq2_toilet",
         Hh_electric  ="P3H_2dq3_electric",
         Hh_cooking_fuel  ="P3H_2dq4_cookfuel",
         Hh_cooking_tech ="P3H_2dq5_cooktech",
         # land_ctrls     
         Hh_own_in = "P3H_2eq1_houses_in",                   
         Hh_own_out = "P3H_2eq2_houses_out",
         # outcomes
         Hh_clear = "P3H_3jq1_forestclearing",
         Hh_clear_total1 = "P3H_3jq2_area_cleared1",
         Hh_clear_total2 = "P3H_3jq2_area_cleared2",
         Hh_clear_total3 = "P3H_3jq2_area_cleared3",
         Forest_cash = "P3H_3jq15_forest_cash"
  ) %>%
  mutate(Period = 2,
         Country_code = as.numeric(Country_code),
         Village_Type = as.numeric(Village_Type),
         Latitude = as.character(Latitude),
         Longitude = as.character(Longitude), 
         Year_formed = as.numeric(Year_formed),
         Hh_lived = as.numeric(Hh_lived),
         Hh_own_in = as.numeric(Hh_own_in),
         Hh_own_out = as.numeric(Hh_own_out),
         Hh_clear_total1 = as.numeric(Hh_clear_total1), 
         Hh_clear_total2 = as.numeric(Hh_clear_total2),
         Hh_clear_total3 = as.numeric(Hh_clear_total3),
         Forest_cash = as.numeric(Forest_cash))


voip3 <- voi[!voi %in% c("Distance_min","Distance_km")]

pd <- p1  %>% 
  dplyr::select(all_of(voi))%>% 
  bind_rows(., p2 %>% dplyr::select(all_of(voi))) %>%
  bind_rows(., p3 %>% dplyr::select(all_of(voip3))) #%>% filter(Country_code %in% c(101, 102, 300))


# transform numerical variables
charvars <- c("Period","Year_formed","Hh_born","Hh_lived",
              "Hh_ethnic","Hh_spouse","Hh_spouse_ethnic", "Hh_floor2","Hh_walls2",
              "Hh_roof2","Hh_water","Hh_toilet","Hh_electric","Hh_cooking_fuel",
              "Hh_cooking_tech", "Hh_own_in","Hh_own_out","Hh_clear","Hh_clear_total1" ,
              "Hh_clear_total2",  "Hh_clear_total3")

pd <- pd %>%
  mutate(across(all_of(charvars), as.numeric))

pd[pd == -8] <- NA
pd[pd == -9] <- NA

##### Period 1 (Baseline) #####

p1.member <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                      sheet = "Tbl_Household_Member_1A", na = c("", "-9"))


p1.member.wide <- p1.member %>%
  group_by(Code_form) %>%
  summarise(hh_head_female = Gender[Relation_hh == 1],
            hh_head_age = Age[Relation_hh == 1],
            hh_member = n(),
            hh_children = length(Name_hh[Age<16]),
            hh_elderly  = length(Name_hh[Age>65]), 
            hh_dependency_ratio = (hh_children+hh_elderly)/hh_member,
            hh_head_occup_agric = grepl("Own production|timber|Forestry|Agricult*", 
                                        Main_occupation_desc[Relation_hh == 1])*1) %>%
  ungroup()


# TO DO!!!

p1.hh.land <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                       sheet = "Tbl_Household_Asset_2A", na = c("", "-9"))

p1.hh.land.wide <- p1.hh.land %>%
  #filter(Area_land_use >= 0) %>%
  group_by(Code_form) %>%
  summarise(land_tot_used  = sum(Area_land_use, na.rm = TRUE),
            land_tot_rent = sum(Area_land_rent, na.rm = TRUE),
            land_tot_borrow = sum(Area_land_borrowed, na.rm = TRUE),
            land_forest_used = sum(Area_land_use[Land_type==2], na.rm = TRUE),
            land_forest_rent = sum(Area_land_rent[Land_type==2], na.rm = TRUE),
            land_forest_borrow = sum(Area_land_borrowed[Land_type==2], na.rm = TRUE),
            land_ag_used = sum(Area_land_use[Land_type==1], na.rm = TRUE),
            land_ag_rent = sum(Area_land_rent[Land_type==1], na.rm = TRUE),
            land_ag_borrow = sum(Area_land_borrowed[Land_type==1], na.rm = TRUE),
            # make subtotals by landuse
            land_tot = land_tot_used + land_tot_rent + land_tot_borrow,
            land_forest_tot = land_forest_used + land_forest_rent + land_forest_borrow,
            land_ag_tot = land_ag_used + land_ag_rent + land_ag_borrow,
            # make area shares
            forest_share = land_forest_tot / land_tot,
            ag_share = land_ag_tot / land_tot) %>%
  ungroup()


p1.hh.tenure <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                         sheet = "Tbl_Household_Tenure_2B", na = c("", "-9"))

p1.hh.tenure.wide <- p1.hh.tenure %>%
  mutate(Tenure_security = as.numeric(Tenure_security)) %>%
  filter(Tenure_catg=="1. Land controlled and used by HH (Col. 1 in Table 2A)",
         Area_parcel > 0,
         Tenure_security >= 0) %>%
  group_by(Code_form) %>%
  summarise(tenure_security = weighted.mean(Tenure_security, Area_parcel)) %>%
  ungroup()

p1.hh.assets <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                         sheet = "Tbl_Household_Other_2EQ3", na = c("", "-9", "-8"))

p1.hh.assets.wide <- p1.hh.assets %>%
  group_by(Code_form) %>%
  summarise(owns_tv = ifelse("Television" %in% Type_asset_desc, 1, 0),
            owns_motorcycle = ifelse("Motorcycle" %in% Type_asset_desc, 1, 0),
            owns_cellphone = ifelse("Cell phone" %in% Type_asset_desc, 1, 0),
            total_asset_value = sum(Tot_value)) %>%
  ungroup()


p1.hh.env.income <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                             sheet = "Tbl_Household_Environmental_3I", na = c("", "-9", "-8"))

p1.hh.env.income.wide <- p1.hh.env.income %>%
  group_by(Code_form) %>%
  summarise(income_forest = sum(Income),
            # Sheets contains no differentiation forest/Agric., therefore, total
            income_nonforest = NA) %>%
  ungroup()


p1.hh.business <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                           sheet = "Tbl_Household_Business_3K", na = c("", "-9", "-8"))

p1.hh.business.wide <- p1.hh.business %>%
  group_by(Code_form) %>%
  # only considering first listed business here!
  summarise(business_net_income = sum(Net_income1)) %>%
  ungroup()


p1.hh.salary <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                         sheet = "Tbl_Household_Salary_3L", na = c("", "-9", "-8"))

p1.hh.salary.wide <- p1.hh.salary %>%
  group_by(Code_form) %>%
  summarise(total_salary = sum(Tot_income),
            agric_salary = sum(Tot_income[grepl("own production|timber|forestry|agricult*", 
                                                tolower(Type_work_desc))])) %>%
  ungroup()


p1.hh.inc.other <- read_xls(paste0(datapath, "/1. Phase 1 Final/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208/M2_GCS_REDD_PHASE1_HOUSEHOLD_211208.xls"), 
                            sheet = "Tbl_Household_Miscellaneous_3M", na = c("", "-9", "-8"))

p1.hh.inc.other.wide <- p1.hh.inc.other %>%
  group_by(Code_form) %>%
  summarise(income_other_tot   = sum(Avg_amount),
            income_gov_trans   = sum(Avg_amount[grepl("government", Type_income_desc)]),
            income_remittances = sum(Avg_amount[grepl("Remittances", Type_income_desc)]),
  ) %>%
  ungroup()

p1.hh.wide <- p1.member.wide %>%
  left_join(., p1.hh.land.wide) %>%
  left_join(., p1.hh.tenure.wide) %>%
  left_join(., p1.hh.assets.wide) %>%
  left_join(., p1.hh.env.income.wide) %>%
  left_join(., p1.hh.business.wide) %>%
  left_join(., p1.hh.salary.wide) %>%
  left_join(., p1.hh.inc.other.wide) %>%
  mutate(Period = 0)




##### Period 2 (First follow up) #####

p2.member <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
               sheet = "Tbl_Household_Member_1A", na = c("", "-9"))


p2.member.wide <- p2.member %>%
  group_by(Code_form) %>%
  summarise(hh_head_female = Gender[Relation_hh == 1],
            hh_head_age = Age[Relation_hh == 1],
            hh_member = n(),
            hh_children = length(Name_hh[Age<16]),
            hh_elderly  = length(Name_hh[Age>65]), 
            hh_dependency_ratio = (hh_children+hh_elderly)/hh_member,
            hh_head_occup_agric = grepl("Own production|timber|Forestry|Agricult*", 
                                             Main_occupation_desc[Relation_hh == 1])*1) %>%
  ungroup()

p2.hh.land <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                          sheet = "Tbl_Household_Asset_2A", na = c("", "-9"))

p2.hh.land.wide <- p2.hh.land %>%
  filter(Area_land_use >= 0) %>%
  group_by(Code_form) %>%
  summarise(land_tot_used  = sum(Area_land_use, na.rm = TRUE),
            land_tot_rent = sum(Area_land_rent, na.rm = TRUE),
            land_tot_borrow = sum(Area_land_borrowed, na.rm = TRUE),
            land_forest_used = sum(Area_land_use[Land_type==2], na.rm = TRUE),
            land_forest_rent = sum(Area_land_rent[Land_type==2], na.rm = TRUE),
            land_forest_borrow = sum(Area_land_borrowed[Land_type==2], na.rm = TRUE),
            land_ag_used = sum(Area_land_use[Land_type==1], na.rm = TRUE),
            land_ag_rent = sum(Area_land_rent[Land_type==1], na.rm = TRUE),
            land_ag_borrow = sum(Area_land_borrowed[Land_type==1], na.rm = TRUE),
            # make subtotals by landuse
            land_tot = land_tot_used + land_tot_rent + land_tot_borrow,
            land_forest_tot = land_forest_used + land_forest_rent + land_forest_borrow,
            land_ag_tot = land_ag_used + land_ag_rent + land_ag_borrow,
            # make area shares
            forest_share = land_forest_tot / land_tot,
            ag_share = land_ag_tot / land_tot) %>%
  ungroup()


p2.hh.tenure <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                         sheet = "Tbl_Household_Tenure_2B", na = c("", "-9"))

p2.hh.tenure.wide <- p2.hh.tenure %>%
  mutate(Tenure_security = as.numeric(Tenure_security)) %>%
  filter(Tenure_catg=="1. Land controlled and used by HH (Col. 1 in Table 2A)",
         Area_parcel > 0,
         Tenure_security >= 0) %>%
  group_by(Code_form) %>%
  summarise(tenure_security = weighted.mean(Tenure_security, Area_parcel)) %>%
  ungroup()

p2.hh.assets <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                       sheet = "Tbl_Household_Other_2EQ3", na = c("", "-9", "-8"))

p2.hh.assets.wide <- p2.hh.assets %>%
  group_by(Code_form) %>%
  summarise(owns_tv = ifelse("Television" %in% Type_asset_desc, 1, 0),
            owns_motorcycle = ifelse("Motorcycle" %in% Type_asset_desc, 1, 0),
            owns_cellphone = ifelse("Cell phone" %in% Type_asset_desc, 1, 0),
            total_asset_value = sum(Tot_value)) %>%
  ungroup()


p2.hh.env.income <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                         sheet = "Tbl_Household_Environmental_3HI", na = c("", "-9", "-8"))

p2.hh.env.income.wide <- p2.hh.env.income %>%
  group_by(Code_form) %>%
  summarise(income_forest = sum(Income[Where_collect_desc=="Forest"]),
            income_nonforest = sum(Income[Where_collect_desc=="Non-forest"])) %>%
  ungroup()


p2.hh.business <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                             sheet = "Tbl_Household_Business_3K", na = c("", "-9", "-8"))

p2.hh.business.wide <- p2.hh.business %>%
  group_by(Code_form) %>%
  summarise(business_net_income = sum(Net_income)) %>%
  ungroup()


p2.hh.salary <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                           sheet = "Tbl_Household_Salary_3L", na = c("", "-9", "-8"))

p2.hh.salary.wide <- p2.hh.salary %>%
  group_by(Code_form) %>%
  summarise(total_salary = sum(Tot_income),
            agric_salary = sum(Tot_income[grepl("own production|timber|forestry|agricult*", 
                                                tolower(Type_work_desc))])) %>%
  ungroup()


p2.hh.inc.other <- read_xls(paste0(datapath, "/2. Phase 2 Final/M2_GCS_PHASE2_HOUSEHOLD_170222/M2_GCS_PHASE2_HOUSEHOLD_170222.xls"), 
                         sheet = "Tbl_Household_Miscellaneous_3M", na = c("", "-9"))

p2.hh.inc.other.wide <- p2.hh.inc.other %>%
  group_by(Code_form) %>%
  summarise(income_other_tot   = sum(Avg_amount),
            income_gov_trans   = sum(Avg_amount[grepl("government", Type_income_desc)]),
            income_remittances = sum(Avg_amount[grepl("Remittances", Type_income_desc)]),
  ) %>%
  ungroup()

p2.hh.wide <- p2.member.wide %>%
  left_join(., p2.hh.land.wide) %>%
  left_join(., p2.hh.tenure.wide) %>%
  left_join(., p2.hh.assets.wide) %>%
  left_join(., p2.hh.env.income.wide) %>%
  left_join(., p2.hh.business.wide) %>%
  left_join(., p2.hh.salary.wide) %>%
  left_join(., p2.hh.inc.other.wide) %>%
  mutate(Period = 1)



##### Period 3 (Second follow up) #####

p3.member <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                       sheet = "q1_household_basicinfo", na = c("", "-9", "-8"))

# there are 29 households with two household heads
# for now, I aggregate by checking whether there is
# any female hh leader and take the maximum age of both heads.
# i.e., taking the max()
# similarly, i check whether any hh head is employed in forest sector.

p3.member.wide <- p3.member %>%
  group_by(P3H_code_form) %>%
  summarise(hh_head_female = max(P3H_1aq3_gender[P3H_1aq2_relation == 1]),
            hh_head_age = max(P3H_1aq4_age[P3H_1aq2_relation == 1]),
            hh_member = n(),
            hh_children = length(P3H_1aq1_name[P3H_1aq4_age<16]),
            hh_elderly  = length(P3H_1aq1_name[P3H_1aq4_age>65]), 
            hh_dependency_ratio = (hh_children+hh_elderly)/hh_member,
            hh_head_occup_agric = any(grepl("Own production|timber|Forestry|Agricult*", 
                                        P3H_1aq6_mainliv_text_English[P3H_1aq2_relation == 1]))*1) %>%
  ungroup()

p3.hh.land <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                        sheet = "q2a_household_landasset", na = c("", "-9", "-8"))

p3.hh.land.wide <- p3.hh.land %>%
  filter(!is.na(P3H_2aq1_used_area)) %>%
  group_by(P3H_code_form) %>%
  summarise(land_tot_used  = sum(P3H_2aq1_used_area, na.rm = TRUE),
            land_tot_rent = sum(P3H_2aq2_rent_area, na.rm = TRUE),
            land_tot_borrow = sum(P3H_2aq3_borrow_area, na.rm = TRUE),
            land_forest_used = sum(P3H_2aq1_used_area[P3H_2a_type==2], na.rm = TRUE),
            land_forest_rent = sum(P3H_2aq2_rent_area[P3H_2a_type==2], na.rm = TRUE),
            land_forest_borrow = sum(P3H_2aq3_borrow_area[P3H_2a_type==2], na.rm = TRUE),
            land_ag_used = sum(P3H_2aq1_used_area[P3H_2a_type==1], na.rm = TRUE),
            land_ag_rent = sum(P3H_2aq2_rent_area[P3H_2a_type==1], na.rm = TRUE),
            land_ag_borrow = sum(P3H_2aq3_borrow_area[P3H_2a_type==1], na.rm = TRUE),
            # make subtotals by landuse
            land_tot = land_tot_used + land_tot_rent + land_tot_borrow,
            land_forest_tot = land_forest_used + land_forest_rent + land_forest_borrow,
            land_ag_tot = land_ag_used + land_ag_rent + land_ag_borrow,
            # make area shares
            forest_share = land_forest_tot / land_tot,
            ag_share = land_ag_tot / land_tot) %>%
  ungroup()


p3.hh.tenure <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                          sheet = "q2b_house_tenure_table", na = c("", "-9", "-8"))

p3.hh.tenure.wide <- p3.hh.tenure %>%
  mutate(Tenure_security = as.numeric(P3H_2bq5_security)) %>%
  filter(P3H_2bq1_tenure_cat=="1",
         P3H_2bq3_parcel> 0,
         Tenure_security >= 0) %>%
  group_by(P3H_code_form) %>%
  summarise(tenure_security = weighted.mean(Tenure_security, P3H_2bq3_parcel)) %>%
  ungroup()

p3.hh.assets <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                          sheet = "q2e3_asset_list", na = c("", "-9", "-8"))

# asset codes:
#   3 = Motorbike 
#  12 = cellphone
#  14 = TV

p3.hh.assets.wide <- p3.hh.assets %>%
  group_by(P3H_code_form) %>%
  summarise(owns_tv = ifelse("14" %in% P3H_2eq1_assettype, 1, 0),
            owns_motorcycle = ifelse("3" %in% P3H_2eq1_assettype, 1, 0),
            owns_cellphone = ifelse("12" %in% P3H_2eq1_assettype, 1, 0),
            total_asset_value = sum(P3H_2eq4_totvalue)) %>%
  ungroup()


p3.hh.env.income <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                              sheet = "q3hi_environment_income", na = c("", "-9", "-8"),
                              col_types = c("numeric", "text", "text",  
                                            "numeric","numeric","numeric","text",
                                            "text","text","numeric","numeric","numeric",
                                            "numeric","numeric","numeric","numeric","text","text"))

p3.hh.env.income.wide <- p3.hh.env.income %>%
  group_by(P3H_code_form) %>%
  summarise(income_forest = sum(P3H_3hiq10_net_income[P3H_3hiq1c_collectfrom==1]),
            income_nonforest = sum(P3H_3hiq10_net_income[P3H_3hiq1c_collectfrom==2])) %>%
  ungroup()


p3.hh.business <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                            sheet = 1, na = c("", "-9", "-8"))

p3.hh.business.wide <- p3.hh.business %>%
  group_by(P3H_code_form) %>%
  summarise(business_net_income = sum(P3H_3kq5_net_income1)) %>%
  ungroup()


p3.hh.salary <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                          sheet = "q3l_wages_income", na = c("", "-9", "-8"))


p3.hh.salary.wide <- p3.hh.salary %>%
  group_by(P3H_code_form) %>%
  summarise(total_salary = sum(P3H_3lq6_tot_income)) %>%
  ungroup()


p3.hh.inc.other <- read_xlsx(paste0(datapath, "/7. Phase 3 Final/M2_GCS_PHASE3_HOUSEHOLD_20211208_20220419_20221125.xlsx"),
                             sheet = "q3m_misc_income", na = c("", "-9", "-8"))

p3.hh.inc.other.wide <- p3.hh.inc.other %>%
  group_by(P3H_code_form) %>%
  summarise(income_other_tot   = sum(P3H_3mq3_tot_amount)) %>%
  ungroup()

p3.hh.wide <-  p3.member.wide %>%
  left_join(., p3.hh.land.wide) %>%
  left_join(., p3.hh.tenure.wide) %>%
  left_join(., p3.hh.assets.wide) %>%
  left_join(., p3.hh.env.income.wide) %>%
  left_join(., p3.hh.business.wide) %>%
  left_join(., p3.hh.salary.wide) %>%
  left_join(., p3.hh.inc.other.wide) %>%
  rename(Code_form = "P3H_code_form") %>%
  mutate(Period = 2)


# Merge all Periods
hh.wide <-     p1.hh.wide  %>%
  bind_rows(., p2.hh.wide) %>%
  bind_rows(., p3.hh.wide)

#View(hh.wide)

# merge control variables
hh_pd_full <- left_join(pd, hh.wide, by=c("Code_form", "Period"))

# final formatting of some columns
hh_pd_full <- hh_pd_full %>%
  mutate(Village_Type = ifelse(Village_Type==1, 1, 0),
         Village = gsub("_", " ", Village),
         Hh_electric = ifelse(Hh_electric == 99, NA, Hh_electric),
         Hh_toilet = ifelse(Hh_toilet == 99, NA, Hh_toilet),
         hh_head_female = as.integer(hh_head_female))

# add outcome variable
hh_pd_full$hh_clearing_total <- rowSums(hh_pd_full[,c("Hh_clear_total1", "Hh_clear_total2", "Hh_clear_total3")])
# replace NAs by zero
hh_pd_full$hh_clearing_total[is.na(hh_pd_full$hh_clearing_total)] <- 0

hh_pd_full$Hh_clear_total1 <- NULL
hh_pd_full$Hh_clear_total2 <- NULL
hh_pd_full$Hh_clear_total3 <- NULL


save(hh_pd_full, file = "./data/hh_pd_full.RData")



##### Filter and clean data (TBD) #####

load(file = "./data/hh_pd_full.RData")

ctrl_vars <- c(
  # Personal 
  "Marital_status", "Year_formed", "Hh_born", "Hh_lived",
  "Hh_ethnic", "Hh_spouse", "Hh_spouse_ethnic", 
  "hh_head_female",  "hh_head_age", 
  "hh_member", "hh_children", "hh_elderly", "hh_dependency_ratio", 
  "hh_head_occup_agric",
  # House assets
  "Hh_cooking_tech",  "Hh_floor2", "Hh_walls2",  "Hh_roof2",
  "Hh_water", "Hh_toilet", "Hh_electric", "Hh_cooking_fuel",
  # Other assets
  "owns_tv", "owns_motorcycle", "owns_cellphone", "total_asset_value",
  # Income
  "income_forest", "business_net_income", "total_salary",
  "agric_salary", "income_other_tot", "income_gov_trans",
  # Land assets
  "Hh_own_in", "Hh_own_out", "tenure_security")

outvars_hh <- c(
  # Land use
  "land_tot", "land_forest_tot", "land_ag_tot", "forest_share","ag_share",
  # Clearing
  "Hh_clear", "hh_clearing_total")

# 1) restrict analysis to three countries
# 2) remove outliers in some control variables
# 3) impute missing values as village level means
# 4) replace some NAs by zero, where appropriate

#### Restrict to BRA, IND, PER ####

# Initially, 10200 observations, 131 villages.

# show availability across projects and periods
# to justify focus on three countries
hh_pd_full %>% 
  mutate(Period_desc = case_when(Period == 0 ~ "Baseline",
                                 Period == 1 ~ "Follow-up 1",
                                 Period == 2 ~ "Follow-up 2")) %>%
  group_by(Country_code, Project_code, Period_desc) %>% 
  summarise(n=n()) %>%
  pivot_wider(names_from = Period_desc, values_from = n)


# -> restrict analysis to 8 projects from three countries
hh_pd_filt <- hh_pd_full %>%
  mutate(country_project = paste(Country_code, Project_code, sep = "_")) %>%
  filter(country_project %in% c("101_01","101_02","101_03","101_04",
                                "102_01", "102_03", 
                                "300_05", "300_06"))

hh_pd_filt %>% 
  group_by(Village_Type) %>% 
  summarise(n_vil = length(unique(Village)))

# Now, 6060 observations, 65 villages.  

#### Remove outliers in some control variables ####

# remove outliers (> 20 IQR in some control variable)
skewed_vars <- c("land_tot","total_asset_value")

# remove outliers in skewed covariates
# defined as larger than 20 times the interquartile range within country and period
hh_pd_filt <- hh_pd_filt %>%
  group_by(Country_code, Period) %>%
  # remove outliers
  mutate(across(all_of(skewed_vars), ~ ifelse(. > 20*IQR(., na.rm=T), NA, .))) %>%
  # log transform skewed variables
  #mutate(across(all_of(skewed_vars), ~ log(1 + .))) %>%
  ungroup() %>%
  # remove missing values
  filter(!is.na(land_tot),
         !is.na(forest_share))

# Now, 6009 observations, removed 51 very large land observations or missing outcomes.


# show attrition rates and make balanced panel

wave_1_idx <- intersect(hh_pd_filt %>% filter(Period == 0) %>% pull(Code_form),
                        hh_pd_filt %>% filter(Period == 1) %>% pull(Code_form))

wave_2_idx <- intersect(hh_pd_filt %>% filter(Period == 1) %>% pull(Code_form),
                        hh_pd_filt %>% filter(Period == 2) %>% pull(Code_form))

wave_all_idx <- intersect(wave_1_idx, wave_2_idx)

# 1338 households in all three waves.
# check for differences in missingness across projects
hh_pd_filt %>%
  # get unique ids with country_project
  distinct(Code_form, country_project) %>%
  mutate(w1 = ifelse(Code_form %in% wave_1_idx, 1, 0),
         w2 = ifelse(Code_form %in% wave_2_idx, 1, 0),
         w_all = ifelse(Code_form %in% wave_all_idx, 1, 0)) %>%
  group_by(country_project) %>%
  summarise(n = n(),
            w1 = mean(w1),
            w2 = mean(w2),
            w_all = mean(w_all))

hh_baseline_with_forest <- hh_pd_filt %>%
  filter(Period == 0,
         !is.na(forest_share),
         forest_share > 0) %>%
  pull(Code_form) %>% unique()

# restrict to baseline forest
hh_pd_bal <- hh_pd_filt %>%
  filter(#Code_form %in% hh_baseline_with_forest, 
         Code_form %in% wave_all_idx)

table(hh_pd_bal$Period)
length(unique(hh_pd_bal$Village))

# Now, 1338 observations, 65 villages.

zero_vars <- c("owns_cellphone", "owns_motorcycle", "owns_tv")

# replace missing values of some variables with zero
hh_pd_bal <- hh_pd_bal %>%
  mutate(across(all_of(zero_vars), ~replace_na(., 0)))

# save
save(hh_pd_bal, file = "./data/hh_pd_bal.RData")


# impute missing values as village level means
hh_pd_bal_imp <- hh_pd_bal %>%
  group_by(Village, Period) %>%
  mutate(across(all_of(ctrl_vars), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .))) %>%
  ungroup()

# save
save(hh_pd_bal_imp, file = "./data/hh_pd_bal_imp.RData")

