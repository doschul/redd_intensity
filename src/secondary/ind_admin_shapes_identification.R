# Alternative shapefiles for Indonesia

# 1) Identify new village shapes for Indonesia
# 2) Run GEE analysis with new shapes and import data
# 3) Compare results with self-reported forest cover data

# select new village shapes for Indonesia

library(tidyverse)
library(sf)
library(rgee)
library(stringr)
library(googledrive)
library(ggpubr)
library(corrplot)


ind_shp_path <- "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/idn_adm_bps_20200401_shp"

load("./data/pd.RData")

ind_l4 <- st_read(paste0(ind_shp_path, "/idn_admbnda_adm4_bps_20200401.shp")) %>%
  filter(grepl("Kalimantan", ADM1_EN)) %>%
  st_make_valid()


# fix one name
ind_l4$ADM4_EN[ind_l4$ADM4_EN == "Long Laay"] <- "Long Laai"

# paste gcs villages that are in shp files
paste(gcs_villages[gcs_villages %in% shp_files$Village], collapse = ", ")




# there are 41 GCS villages in Indonesia
# 25 of these have been matched with shape data
# 29 are present in official admin shape data

# get REDD village names
gcs_villages <- sort(unique(pd$Village[pd$Country_code=="300"]))

# intersect with shp files
shp_jnd <- st_join(ind_l4, shp_files)

# subset shapes that have some union and View
intersecting_shps <- shp_jnd[!is.na(shp_jnd$Village),]


# make table of village names and in which dataset they occur
df_gcs <- data.frame(Village = gcs_villages) %>%
  mutate(shp_files = (Village %in% shp_files$Village)*1,
         ind_l4_int = (Village %in% intersecting_shps$ADM4_EN)*1,
         ind_l4_all = (Village %in% ind_l4$ADM4_EN)*1)



gcs_shp_adm <- ind_l4[ind_l4$ADM4_EN %in% gcs_villages,]

# check for duplicates
gcs_shp_adm$ADM4_EN[which(duplicated(gcs_shp_adm$ADM4_EN))]

# ind_l4_name | shp_name
#_____________|__________
# Long Laay   | Long Laai -> checked
# Sungai Besar occurs five times, is certainly 
# not PCode ID6303070023, ID6372032004, ID6108040012 (urban or very off track)
# could be ID6112010004, but most likely ID6106060006
# most likely ID6112010004, which is adjacent to ID6112010003

# Panaan most certainly ID6405010004, NOT ID6309081006.

# Sungai Jawi unlikely ID6106060008 m ID6171031005 (urban)
# most likely ID6112010003

exclude_ids <- c("ID6303070023", "ID6372032004", "ID6108040012", "ID6112010004",
                 "ID6309081006", "ID6106060008", "ID6171031005")

# exclude from shapefile
gcs_shp_adm <- gcs_shp_adm[!gcs_shp_adm$ADM4_PCODE %in% exclude_ids,]

# add credibility column based on intersection with official shapefile
gcs_shp_adm <- gcs_shp_adm %>%
  mutate(credibility = (ADM4_EN %in% intersecting_shps$ADM4_EN)*1) %>%
  select(ADM4_EN, credibility, geometry)

# show villages remaining without shape
gcs_villages[!gcs_villages %in% gcs_shp_adm$ADM4_EN]


# save as gpkg
st_write(gcs_shp_adm, "./indonesia_villages.gpkg")


# save as shp

gcs_shp_adm$name_clean <- iconv(gcs_shp_adm$ADM4_EN, from = 'UTF-8', 
                                to = 'ASCII//TRANSLIT') %>%
  tolower() %>%
  gsub(" ", "_", .) %>%
  gsub("-", "_", .)

st_write(gcs_shp_adm, "./ind_vil_new.shp", overwrite = TRUE)

##### GEE analysis with new shapes #####
# - The extraction part was done in GEE directly by uploading the new shapefile.

# download data
ee_Initialize(drive = TRUE)

# downlaod all files from google drive
drive_auth()

#folder link to id
jp_folder = "https://drive.google.com/drive/u/0/folders/1Xkm7dshJvHL2106WI2hNUNsQZq4eZ1Q7"
folder_id = drive_get(as_id(jp_folder))

#find files in folder
files = drive_ls(folder_id)

files <- files[grepl("ind_new", files$name),]

#loop dirs and download files inside them
for (i in seq_along(files$name)) {
  
  #fails if already exists
  try({
    drive_download(
      as_id(files$id[i]),
      path = str_c("./data/gee_jrc/", files$name[i])
    )
  })
}

# read files
jrc_files <- list.files("./data/gee_jrc/", 
                        pattern = "csv$", 
                        full.names = T) %>%
  subset(grepl("ind_new", .))

jrc_dat <- do.call(rbind, lapply(jrc_files, function(x) cbind(read_csv(x), name=tail(strsplit(x, "/")[[1]], 1))))

jrc_dat$year <- as.integer(substr(jrc_dat$name, 25, 28))
jrc_dat$`system:index` <- NULL
jrc_dat$.geo <- NULL
jrc_dat$name <- NULL

save(jrc_dat, file = "./data/jrc_dat_ind_new.RData")


#### RS data comparison with SR data #####

load("./data/jrc_dat_ind_new.RData")

# merge with aggregated village data
load("./data/vil_treat_agg.RData")
load("./data/pd.RData")

# aggregate self-reported data to village level
self_reported_vil <- pd %>%
  filter(forest_share > 0) %>%
  #filter(Village %in% project_df$Village) %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018)) %>%
  group_by(year, Village) %>%
  summarise(sr_area_ha = sum(land_tot, na.rm = TRUE),
            sr_forest_ha = sum(land_forest_tot, na.rm = TRUE),
            sr_agric_ha = sum(land_ag_tot, na.rm = TRUE),
            sr_forest_perc = sr_forest_ha / sr_area_ha,
            sr_ag_perc = sr_agric_ha / sr_area_ha,
            sr_def_bin_perc = mean(Hh_clear, na.rm = TRUE),
            sr_def_ha = sum(hh_clearing_total, na.rm = TRUE),
            sf_def_perc = sr_def_ha / sr_area_ha) %>%
  ungroup()

undist_fc_2000 <- jrc_dat %>%
  filter(year == 2000) %>%
  mutate(jrc_fc2000_ha = UndisturbedForest / 10000) %>%
  select(ADM4_EN, jrc_fc2000_ha)

# merge
vil_ind <- self_reported_vil %>%
  left_join(jrc_dat, by = c("Village" = "ADM4_EN", "year"= "year")) %>%
  left_join(., undist_fc_2000, by = c("Village" = "ADM4_EN")) %>%
  filter(!is.na(UndisturbedForest)) %>%
  # calculate relative loss variables
  mutate(
    jrc_perc_UndisturbedForest = (UndisturbedForest / 10000) / jrc_fc2000_ha,
    jrc_perc_ForestDegradation = (ForestDegradation / 10000) / jrc_fc2000_ha,
    jrc_perc_DirectDeforestation = (DirectDeforestation / 10000) / jrc_fc2000_ha,
    jrc_perc_DeforAfterDegrad = (DeforAfterDegrad / 10000) / jrc_fc2000_ha,
    jrc_perc_DefTotal = (DirectDeforestation + DeforAfterDegrad) / 10000 / jrc_fc2000_ha
  )


p.qq.sr.fc.ind <- qqplot_fun(vil_ind, 
                             "sr_forest_perc", "jrc_perc_UndisturbedForest", 
                             title = "Indonesia new shapes: SR vs. JRC forest cover (%)")



