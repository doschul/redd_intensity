# Goal: compile village level dataset
# Input: village polygons, village level survey

##### SETUP #####

#### Working environment ####

rm(list = ls())

setwd("C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/research/redd_intensity")

shape_path <- "C:/Users/DaSchulz/OneDrive - European Forest Institute/Dokumente/projects/cifor/1. Research/7. M2 Maps/A.4. GIS Files/GCS_M2_ph3_VillageData_20180531/shp/"

# ToDo: Fix Pilao Poente shapefile and rerun analysis!


#### Libraries ####

library(tidyverse)
library(sf)
library(rgee)
library(stringr)
library(googledrive)

#### helper functions ####

# Function to check and rename the column
rename_village_na <- function(df) {
  if ("Village_na" %in% colnames(df)) {
    colnames(df)[colnames(df) == "Village_na"] <- "Village"
  }
  return(df)
}

##### 1 load and clean shapes ####

# import village shape files
shp_files <- list.files(shape_path, pattern = "pol_tot.shp$", 
                        full.names = T) %>%
  lapply(., st_read) %>%
  # Apply the function to each dataframe in the list
  lapply(., rename_village_na) %>%
  # Select columns from each dataframe in the list and convert them to character
  lapply(., function(df) {
    selected <- df[, c("Village", "Village_Ty")]
    selected$Village_Ty <- as.character(selected$Village_Ty)
    return(selected)
  }) %>%
  do.call(rbind, .)

#pilao_poente <- shp_files[shp_files$Village=="Pilão Poente",] %>%
#  st_make_valid() # remove self-intersections


#st_is_valid(pilao_poente, reason = TRUE)
sf_use_s2(FALSE)


shp_files <- sf::st_make_valid(shp_files)
shp_files <- shp_files[st_is_valid(shp_files),]
shp_files$area_km2 <- as.numeric(st_area(shp_files)/ 1000000)

sf_use_s2(TRUE)

# clean some village names
shp_files <- shp_files %>%
  mutate(Village = case_when(Village == "P.A. Aleluia" ~ "Aleluia",
                             Village == "Serengkah Kiri" ~ "Serengkah kiri",
                             Village == "Tewang Baringin" ~ "Tewang Beringin",
                             Village == "Long Noran / Norah" ~ "Long Noran",
                             Village == "338" ~ "Km 338",
                             Village == "Vicinal do Pão Doce (331)" ~ "Vicinal do Pão Doce (km 331)",
                             #Village == "338" ~ "Pilão Poente",
                             Village == "Vicinal 3 barracas" ~ "Vicinal Três Barracas",
                             TRUE ~ Village))

# clean up two specific villages
shp_files$Village[shp_files$Village == "Nova Esperança"] <- c("Nova Esperança-AC", "Nova Esperança-MT")

shp_files$vil_clean <- iconv(shp_files$Village, from = 'UTF-8', to = 'ASCII//TRANSLIT') %>%
  tolower() %>%
  gsub(" ", "_", .) %>%
  gsub("-", "_", .)

# fix village types
shp_files <- shp_files %>%
  mutate(Village_Ty = case_when(Village_Ty == "1" ~ "Intervention",
                                Village_Ty == "2" ~ "Control",
                                TRUE ~ Village_Ty))



# save cleaned shapefiles

cifor_vil_poly <- shp_files[,"vil_clean"]

#sf::st_write(cifor_vil_poly, "cifor_vil_poly.shp")

# outer buffer function
make_buffer <- function(df, dist=100, gap = 0) {
  original_crs <- st_crs(df)
  
  df <- df %>%
    st_transform(32722)
  
  origin_union <- st_union(df) %>%
    st_make_valid()
  
  
  if (gap == 0) {
    # make buffer and remove original inner shapes (one by one)
    buff <- lapply(1:nrow(df), function(x){
      st_difference(st_buffer(df[x,], 
                              dist=dist),
                    df[x,]) %>%
        # also remove entirety of original shapes
        st_difference(., origin_union)
    }) %>% do.call(rbind.data.frame,.)
  }
  
  # if gap is specified, first buffer the original shapes 
  # and then remove the inner shapes and the gap
  if (gap > 0) {
    gaps <- st_buffer(df, dist=gap)
    
    gap_union <- st_union(gaps) %>%
      st_make_valid()
    
    buff <- lapply(1:nrow(gaps), function(x){
      st_difference(st_buffer(gaps[x,], 
                              dist=dist),
                    gaps[x,]) %>%
        st_difference(., gap_union)
    }) %>% do.call(rbind.data.frame,.)
  }
  
  # assign original crs
  buff <- buff %>%
    st_transform(original_crs)
  
  return(buff[,1])
}

# make buffer polygons
b_0_100 <- make_buffer(cifor_vil_poly, dist = 100, gap = 0)
b_0_500 <- make_buffer(cifor_vil_poly, dist = 500, gap = 0)
b_0_1000 <- make_buffer(cifor_vil_poly, dist = 1000, gap = 0)
b_0_5000 <- make_buffer(cifor_vil_poly, dist = 5000, gap = 0)

b_100_500 <- make_buffer(cifor_vil_poly, dist = 500, gap = 100)
b_500_1000 <- make_buffer(cifor_vil_poly, dist = 1000, gap = 500)
b_1000_5000 <- make_buffer(cifor_vil_poly, dist = 5000, gap = 1000)


# save buffer polygons
sf::st_write(b_0_100, "./data/raw/vil_shp/village_buffer/b_0_100.shp", append = F)
sf::st_write(b_0_500, "./data/raw/vil_shp/village_buffer/b_0_500.shp", append = F)
sf::st_write(b_0_1000, "./data/raw/vil_shp/village_buffer/b_0_1000.shp", append = F)
sf::st_write(b_0_5000, "./data/raw/vil_shp/village_buffer/b_0_5000.shp", append = F)
sf::st_write(b_100_500, "./data/raw/vil_shp/village_buffer/b_100_500.shp", append = F)
sf::st_write(b_500_1000, "./data/raw/vil_shp/village_buffer/b_500_1000.shp", append = F)
sf::st_write(b_1000_5000, "./data/raw/vil_shp/village_buffer/b_1000_5000.shp", append = F)

# combine buffers into one shapefile
buffers <- rbind(b_0_100, b_0_500, b_0_1000, b_0_5000, b_100_500, b_500_1000, b_1000_5000)

# add buffer type
buffers$buffer <- rep(c("b_0_100", "b_0_500","b_0_1000","b_0_5000",
                        "b_100_500","b_500_1000","b_1000_5000"), 
                      each = nrow(cifor_vil_poly))

sf::st_write(buffers, "./data/raw/vil_shp/village_buffer/all_buffers.shp", append = F)


# count observations per shapefile
n_obs <- list.files(shape_path, pattern = "pol_tot.shp$", 
                    full.names = T) %>%
  lapply(., st_read) %>%
  # filter out invalid geometries
  # lapply(., function(x) {
  #   x <- st_make_valid(x)
  #   x <- x[st_is_valid(x),]
  #   return(x)
  # }) %>%
  # count number of rows
  lapply(., nrow) %>%
  unlist()

project_names <- list.files(shape_path, pattern = "pol_tot.shp$", 
                            full.names = F) %>%
  # extract project names
  gsub("_vil_pol_tot.shp", "", .)

project_df <- data.frame(Village = shp_files$Village,
                         vil_clean = shp_files$vil_clean,
                         Village_Ty = shp_files$Village_Ty,
                         project = rep(project_names, n_obs),
                         area_km2 = shp_files$area_km2)  

# save project_df
save(project_df, file = "./data/rdat/project_df.RData")

##### 2 GEE data #####

#### 2a) Hansen forest loss data ####

#ee_Initialize()
ee_Initialize()

# load shapefile
ee_sites <- rgee::sf_as_ee(cifor_vil_poly)


# Load the Hansen Global Forest Change dataset
hansen <- ee$Image("UMD/hansen/global_forest_change_2023_v1_11")

# Extract forest cover for the year 2000 (treecover2000 band) 
forest2000 <- hansen$select("treecover2000")

# Apply the FAO threshold (tree cover ≥ 10%)
forest2000_threshold <- forest2000$gte(10)

# make binary forest mask
forest_2000_mask <- forest2000$mask(forest2000_threshold)

# get lossyear layer
forest_loss <- hansen$select("lossyear")

# Create a new image with each band representing one year

# Create a list of annual loss images, masked by forest cover in 2000
years <- 2000:2023
gfc_bands <- lapply(years, function(year) {
  # Select loss pixels for the specific year (year - 2000 since loss year band starts at 1 for 2001)
  loss_year <- hansen$select("lossyear")$eq(year - 2000)
  
  # Mask the loss image to include only areas that were forested in 2000
  masked_loss <- loss_year$updateMask(forest_2000_mask)
  
  # Rename the band to reflect the specific year
  masked_loss$rename(paste0('gfc_', year))
})


# Combine all the bands into a single image
gfc_image <- do.call(ee$Image$cat, gfc_bands)

# add forest mask as band
gfc_image <- gfc_image$addBands(forest2000_threshold)

# show bandnames
gfc_image$getInfo()


# extract
gfc_extract <- ee_extract(
  x = gfc_image,
  y = cifor_vil_poly,
  fun = ee$Reducer$sum(),
  scale = 30,
  sf = FALSE
)

gfc_extract$gfc_2000 <- NULL
gfc_extract <- left_join(project_df[,c("vil_clean", "area_km2")], 
                         gfc_extract, by = "vil_clean")


# to long format
gfc_long <- gfc_extract %>%
  pivot_longer(cols = starts_with("gfc_"),
               names_to = "year",
               names_prefix = "gfc_",
               values_to = "gfc_def_pix") %>%
  mutate(year = as.integer(year),
         area_ha = area_km2 * 100)

# Calculate cumulative forest loss and deforestation rates
gfc_long <- gfc_long %>%
  # pixel to hectare
  mutate(gfc_fc2000_ha = treecover2000 * 0.09,
         gfc_def_ha = gfc_def_pix * 0.09, 
         gfc_fc2000_perc = gfc_fc2000_ha / area_ha) %>%
  group_by(vil_clean) %>%
  arrange(year) %>%
  mutate(
    gfc_cumdef_ha = cumsum(gfc_def_ha),
    gfc_remaining_forest_ha = gfc_fc2000_ha - gfc_cumdef_ha,
    gfc_remaining_forest_perc = gfc_remaining_forest_ha / gfc_fc2000_ha,
    gfc_def_perc_tot_area = (gfc_def_ha / area_ha),
    gfc_def_perc_fc2000 = (gfc_def_ha / gfc_fc2000_ha),
    gfc_def_perc_prev_year = (gfc_def_ha / (gfc_def_ha + gfc_remaining_forest_ha)),
    gfc_def_cum_fc2000 = gfc_cumdef_ha / gfc_fc2000_ha
  ) %>%
  ungroup()



# save
save(gfc_long, file = "./data/rdat/gfc_long.RData")

#### 2b) JRC TDF data ####

ee_Initialize(drive = TRUE)


#*** 1) Load your study area as a feature collection. For example:
StudyArea <- rgee::sf_as_ee(shp_files)
AOIname <- 'cifor' # Replace with the explicit name of the study area

#*** 2) Indicate the timeframe for extracting statistics
StartYear <- 2000
EndYear <- 2020

#*** 3) Specify the Google Drive folder for exporting results
GoogleDriveFolder <- 'gee_jrc' # Change as needed

Transition <- ee$ImageCollection('projects/JRC/TMF/v1_2023/TransitionMap_Subtypes')$mosaic()
DeforestationYear <- ee$ImageCollection('projects/JRC/TMF/v1_2023/DeforestationYear')$mosaic()
Intensity <- ee$ImageCollection('projects/JRC/TMF/v1_2023/Intensity')$mosaic()
Duration <- ee$ImageCollection('projects/JRC/TMF/v1_2023/Duration')$mosaic()
DegradationYear <- ee$ImageCollection('projects/JRC/TMF/v1_2023/DegradationYear')$mosaic()
AnnualChanges <- ee$ImageCollection('projects/JRC/TMF/v1_2023/AnnualChanges')$mosaic()

#* Deforestation
OldDeforestation <- (Transition$gte(41)$And(Transition$lte(42)))$Or(Transition$gte(65)$And(Transition$lte(66)))
RecentDeforestation <- (Transition$gte(51)$And(Transition$lte(53)))$Or(Transition$eq(67)$And(DeforestationYear$lt(2021)))$Or(Transition$eq(67)$And(DeforestationYear$gte(2021))$And(Intensity$gte(10)))
DeforestationToWater <- Transition$gte(73)$And(Transition$lte(74))
DeforestationToPlantations <- Transition$gte(82)$And(Transition$lte(86))

#* Forest degradation
ShortDurationDegradation <- (Transition$gte(21)$And(Transition$lte(22)))$Or(Transition$gte(61)$And(Transition$lte(62))$And(Duration$lte(365)))$Or(Transition$eq(54)$Or(Transition$eq(67)$And(DegradationYear$gte(2021))$And(Intensity$lt(10))))
LongDurationDegradation <- (Transition$gte(23)$And(Transition$lte(26)))$Or(Transition$gte(61)$And(Transition$lte(62))$And(Duration$gt(365)))

#* Vegetation regrowth
Regrowth <- (Transition$gte(31)$And(Transition$lte(33)))$Or(Transition$gte(63)$And(Transition$lte(64)))

# Loop through years
for (i in StartYear:(EndYear+1)) {
  
  cat(i, '\n')
  
  j <- i - 1
  year <- paste0('Dec', i)
  year_minus1 <- paste0('Dec', j)
  
  AnnualChangesYear <- AnnualChanges$select(year)
  AnnualChangesYearMinus1 <- AnnualChanges$select(year_minus1)
  
  #*** REMAINING UNDISTURBED FOREST
  Class10 <- AnnualChangesYear$eq(1)
  
  #*** NEW DEGRADATION
  newdegradation <- AnnualChangesYearMinus1$eq(1)$And(AnnualChangesYear$eq(2))
  Class21 <- newdegradation$And(ShortDurationDegradation)
  Class22 <- newdegradation$And(LongDurationDegradation)
  Class23 <- newdegradation$And(OldDeforestation$Or(RecentDeforestation)$Or(Regrowth))
  totalDegradation <- Class21$Or(Class22)$Or(Class23)
  
  #*** NEW DEFORESTATION
  newdefordirect <- AnnualChangesYearMinus1$eq(1)$And(AnnualChangesYear$eq(3))
  Class31 <- newdefordirect$And(RecentDeforestation$Or(OldDeforestation))
  Class32 <- newdefordirect$And(Regrowth)
  Class33 <- newdefordirect$And(DeforestationToPlantations)
  
  newdeforwater <- AnnualChangesYearMinus1$eq(1)$And(AnnualChangesYear$eq(5))
  Class34 <- DeforestationToWater$And(newdefordirect$Or(newdeforwater))
  
  # After degradation
  newdeforafterdegrad <- AnnualChangesYearMinus1$eq(2)$And(AnnualChangesYear$eq(3))
  Class35 <- newdeforafterdegrad$And(RecentDeforestation$Or(OldDeforestation))
  Class36 <- newdeforafterdegrad$And(Regrowth)
  
  totalDeforestation <- Class31$Or(Class32)$Or(Class33)$Or(Class34)$Or(Class35)$Or(Class36)
  totalDirectDeforestation <- Class31$Or(Class32)$Or(Class33)$Or(Class34)
  
  #*** NEW DISTURBANCE
  Disturbance <- newdegradation$Or(newdefordirect)
  
  #*** NEW REGROWTH
  newregrowth <- AnnualChangesYearMinus1$eq(3)$And(AnnualChangesYear$eq(4))
  Class41 <- newregrowth$And(Regrowth)
  
  # Choose to extract main or detailed classes
  AllClasses <- ee$Image$cat(
    Class10$rename('UndisturbedForest'),
    totalDegradation$rename('ForestDegradation'),
    totalDirectDeforestation$rename('DirectDeforestation'),
    newdeforafterdegrad$rename('DeforAfterDegrad'),
    Class41$rename('Regrowth')
  )
  
  # Function for reducing regions
  LOOPsamples <- function(feature) {
    vals <- AllClasses$multiply(ee$Image$pixelArea())$reduceRegion(
      reducer = ee$Reducer$sum(),
      geometry = feature$geometry(),
      scale = 30,
      maxPixels = 5e9
    )
    ee$Feature(NULL, vals)$copyProperties(feature, feature$propertyNames())
  }
  
  LOOPresult2 <- StudyArea$map(LOOPsamples)
  
  # Export results
  
  task <- ee_table_to_drive(
    collection = LOOPresult2,
    description = paste0('AnnualChange_', AOIname, '_', year),
    folder = GoogleDriveFolder,
    fileNamePrefix = paste0('AnnualChange_', AOIname, '_', year)
  )
  
  task$start()
  
  ee_monitoring(task)
  
}

# downlaod all files from google drive
drive_auth()

#folder link to id
jp_folder = "https://drive.google.com/drive/u/0/folders/1Xkm7dshJvHL2106WI2hNUNsQZq4eZ1Q7"
folder_id = drive_get(as_id(jp_folder))

#find files in folder
files = drive_ls(folder_id)

# consider only buffer files
files <- files %>% filter(grepl("buffers", name))

#loop dirs and download files inside them
for (i in seq_along(files$name)) {
  
  #fails if already exists
  try({
    drive_download(
      as_id(files$id[i]),
      path = str_c("./data/raw/gee_jrc/", files$name[i])
    )
  })
}

# read files
jrc_files <- list.files("./data/raw/gee_jrc/", 
                        pattern = "csv$", 
                        full.names = T)

jrc_dat <- do.call(rbind, lapply(jrc_files, function(x) cbind(read_csv(x), name=tail(strsplit(x, "/")[[1]], 1))))

jrc_dat$year <- as.integer(substr(jrc_dat$name, 23, 26))
jrc_dat$`system:index` <- NULL
jrc_dat$.geo <- NULL
jrc_dat$name <- NULL

save(jrc_dat, file = "./data/rdat/jrc_dat.RData")


# read files
jrc_buffer_files <- list.files("./data/raw/gee_jrc/", 
                        pattern = "csv$", 
                        full.names = T) %>%
  str_subset("buffers")

jrc_buffer_dat <- do.call(rbind, lapply(jrc_buffer_files, function(x) cbind(read_csv(x), name=tail(strsplit(x, "/")[[1]], 1))))

jrc_buffer_dat$year <- as.integer(substr(jrc_buffer_dat$name, 31, 34))
jrc_buffer_dat$`system:index` <- NULL
jrc_buffer_dat$.geo <- NULL
jrc_buffer_dat$name <- NULL


buf_undist_fc_2000 <- jrc_buffer_dat %>%
  filter(year == 2000) %>%
  mutate(jrc_fc2000_ha = UndisturbedForest / 10000) %>%
  select(vil_clean, buffer, jrc_fc2000_ha)

jrc_buffer_wide <-  jrc_buffer_dat %>%
  left_join(., buf_undist_fc_2000, by = c("vil_clean", "buffer")) %>%
  # calculate relative loss variables
  mutate(
    jrc_fc = (UndisturbedForest / 10000) / jrc_fc2000_ha,
    jrc_deg = (ForestDegradation / 10000) / jrc_fc2000_ha,
    jrc_def = (DirectDeforestation / 10000) / jrc_fc2000_ha,
    jrc_degdef = (DeforAfterDegrad / 10000) / jrc_fc2000_ha,
    jrc_deftot = (DirectDeforestation + DeforAfterDegrad) / 10000 / jrc_fc2000_ha
  )  %>%
  # remove original columns
  select(-jrc_fc2000_ha, -UndisturbedForest, -ForestDegradation, 
         -DirectDeforestation, -DeforAfterDegrad, -Regrowth) %>% 
  # transform to wide
  pivot_wider(names_from = "buffer",
              values_from = c("jrc_fc", 
                              "jrc_deg", 
                              "jrc_def", 
                              "jrc_degdef", 
                              "jrc_deftot"))

library(data.table)

# turn into data.table
jrc_buffer_wide <- as.data.table(jrc_buffer_wide)

# sort by village and year
jrc_buffer_wide <- jrc_buffer_wide[order(vil_clean, year)]

nm1 <- grep("jrc", colnames(jrc_buffer_wide), value=TRUE)
nm2 <- paste("t1", nm1, sep="_")

jrc_buffer_wide_lag <- jrc_buffer_wide[, (nm2) :=  shift(.SD), by=vil_clean, .SDcols=nm1]

# replace missing fc-column values with 1, all others with 0
jrc_buffer_wide_lag <- jrc_buffer_wide_lag %>%
  mutate_at(vars(starts_with("t1_jrc_f")), ~replace(., is.na(.), 1)) %>%
  mutate_at(vars(starts_with("t1_jrc_d")), ~replace(., is.na(.), 0))


save(jrc_buffer_wide_lag, file = "./data/rdat/jrc_buffer_wide_lag.RData")


#### 2c) Village level covariate data ####

vdf <- readxl::read_xlsx("./data/raw/Processed_GCS_village_panel_16Dec16.xlsx")

# Terraclimate precipitation data
tc_prec <- ee$ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") %>%
  ee$ImageCollection$filterDate("2000-01-01", "2022-12-31") %>%
  ee$ImageCollection$map(function(x) x$select("pr")) %>% # Select only precipitation bands
  ee$ImageCollection$toBands() %>% # from imagecollection to image
  ee$Image$rename(paste0("prec_", rep(c(2000:2022), each = 12), "_", c(1:12))) # rename the bands of an image

tc_tmmx <- ee$ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") %>%
  ee$ImageCollection$filterDate("2000-01-01", "2022-12-31") %>%
  ee$ImageCollection$map(function(x) x$select("tmmx")) %>% # Select only precipitation bands
  ee$ImageCollection$toBands() %>% # from imagecollection to image
  ee$Image$rename(paste0("tmmx_", rep(c(2000:2022), each = 12), "_", c(1:12))) # rename the bands of an image

tc_wd <- ee$ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") %>%
  ee$ImageCollection$filterDate("2000-01-01", "2022-12-31") %>%
  ee$ImageCollection$map(function(x) x$select("def")) %>% # Select only water deficit bands
  ee$ImageCollection$toBands() %>% # from imagecollection to image
  ee$Image$rename(paste0("wd_", rep(c(2000:2022), each = 12), "_", c(1:12))) # rename the bands of an image


# extract
ee_tc_prec <- ee_extract(x = tc_prec, y = cifor_vil_poly, fun = ee$Reducer$sum(),
                         scale = 500,sf = FALSE)
ee_tc_tmmx <- ee_extract(x = tc_tmmx, y = cifor_vil_poly, fun = ee$Reducer$mean(),
                         scale = 500,sf = FALSE)
ee_tc_wd  <- ee_extract(x = tc_wd,  y = cifor_vil_poly, fun = ee$Reducer$mean(),
                         scale = 500,sf = FALSE)

# join
ee_long <- ee_tc_prec %>%
  left_join(., ee_tc_tmmx, by = "vil_clean") %>%
  left_join(., ee_tc_wd, by = "vil_clean") %>%
  pivot_longer(cols = -"vil_clean",
               names_to = c("var", "year", "month"),
               names_pattern = "(.*)_(.*)_(.*)",
               values_to = "value")

terraclimate_wide <- ee_long %>%
  mutate(quarter = case_when(month %in% 1:3 ~ 1,
                             month %in% 4:6 ~ 2,
                             month %in% 7:9 ~ 3,
                             month %in% 10:12 ~ 4)) %>%
  group_by(vil_clean, var, year) %>%
  summarise(tot_sum  = sum(value),
            tot_mean = mean(value),
            q1_sum  = sum(value[quarter == 1]),
            q1_mean = mean(value[quarter == 1]),
            q2_sum  = sum(value[quarter == 2]),
            q2_mean = mean(value[quarter == 2]),
            q3_sum  = sum(value[quarter == 3]),
            q3_mean = mean(value[quarter == 3]),
            q4_sum  = sum(value[quarter == 4]),
            q4_mean = mean(value[quarter == 4])) %>%
  ungroup() %>%
  pivot_wider(names_from = c("var"),
              values_from = c("tot_sum", "tot_mean", 
                              "q1_sum", "q1_mean", 
                              "q2_sum", "q2_mean", 
                              "q3_sum", "q3_mean", 
                              "q4_sum", "q4_mean")) %>%
  mutate(year = as.integer(year))

# save
save(terraclimate_wide, file = "./data/rdat/terraclimate_wide.RData")

#### population data ####

gee_pop <- ee$ImageCollection("CIESIN/GPWv411/GPW_Population_Count") %>%
  ee$ImageCollection$filterDate("2000-01-01", "2020-12-31") %>%
  ee$ImageCollection$toBands() # from imagecollection to image

# extract
ee_pop_dens <- ee_extract(x = gee_pop, y = cifor_vil_poly, fun = ee$Reducer$mean(),
                     scale = 500,sf = FALSE)
ee_pop <- ee_extract(x = gee_pop, y = cifor_vil_poly, fun = ee$Reducer$sum(),
                          scale = 500,sf = FALSE)

# rename columns
names(ee_pop) <- c("vil_clean", "pop2000", "pop2005", "pop2010", "pop2015", "pop2020")
# rename columns
names(ee_pop_dens) <- c("vil_clean", "popdens2000", "popdens2005", "popdens2010", 
                        "popdens2015", "popdens2020")

# long format and fill missing years
ee_pop_long <- ee_pop %>%
  pivot_longer(cols = -"vil_clean",
               names_to = "year",
               values_to = "pop") %>%
  mutate(year = as.integer(gsub("pop", "", year))) %>%
  complete(vil_clean, year = 2000:2022) %>%
  fill(pop, .direction = "down") %>%
  fill(pop, .direction = "up")

# long format and fill missing years
ee_popdens_long <- ee_pop_dens %>%
  pivot_longer(cols = -"vil_clean",
               names_to = "year",
               values_to = "popdens") %>%
  mutate(year = as.integer(gsub("popdens", "", year))) %>%
  complete(vil_clean, year = 2000:2022) %>%
  fill(popdens, .direction = "down") %>%
  fill(popdens, .direction = "up")

# merge
ee_pop_long <- ee_pop_long %>%
  left_join(., ee_popdens_long, by = c("vil_clean", "year"))

# save
save(ee_pop_long, file = "./data/rdat/population.RData")

#### 2d) merge data into long format ####

rm(list = ls())

#load("./data/forest_loss.RData")
load("./data/rdat/project_df.RData")
load("./data/rdat/gfc_long.RData")
load("./data/rdat/jrc_dat.RData")
load("./data/rdat/jrc_buffer_wide_lag.RData")
load("./data/rdat/hh_pd.RData")
load("./data/rdat/vil_treat_agg.RData")
load("./data/rdat/terraclimate_wide.RData")
load("./data/rdat/population.RData")


# aggregate self-reported data to village level
self_reported_vil_hh <- hh_pd %>%
  filter(Village %in% project_df$Village) %>%
  mutate(year = case_when(Period == 0 ~ 2010,
                          Period == 1 ~ 2014,
                          Period == 2 ~ 2018)) %>%
  group_by(year, Village) %>%
  summarise(sr_area_ha = sum(land_tot, na.rm = TRUE),
            sr_forest_ha = sum(land_forest_tot, na.rm = TRUE),
            sr_agric_ha = sum(land_ag_tot, na.rm = TRUE),
            sr_forest_perc = mean(forest_share, na.rm = TRUE),
            sr_ag_perc = mean(ag_share, na.rm = TRUE),
            sr_def_bin_perc = mean(Hh_clear, na.rm = TRUE),
            sr_def_ha = sum(hh_clearing_total, na.rm = TRUE),
            sf_def_perc = sr_def_ha / sr_area_ha) %>%
  ungroup()

undist_fc_2000 <- jrc_dat %>%
  filter(year == 2000) %>%
  mutate(jrc_fc2000_ha = UndisturbedForest / 10000) %>%
  select(vil_clean, jrc_fc2000_ha)

vil_pd <- project_df %>%
  left_join(., gfc_long %>% filter(year <= 2020), by = "vil_clean") %>%
  left_join(., jrc_dat, by = c("vil_clean", "year")) %>%
  left_join(., undist_fc_2000, by = "vil_clean") %>%
  left_join(., self_reported_vil_hh, by = c("Village", "year")) %>%
  left_join(., vil_treat_agg, by = c("Village", "year")) %>%
  left_join(., terraclimate_wide, by = c("vil_clean", "year")) %>%
  left_join(., ee_pop_long, by = c("vil_clean", "year")) %>%
  left_join(., jrc_buffer_wide_lag, by = c("vil_clean", "year")) %>%
  # calculate relative loss variables
  mutate(
    jrc_perc_UndisturbedForest = (UndisturbedForest / 10000) / jrc_fc2000_ha,
    jrc_perc_ForestDegradation = (ForestDegradation / 10000) / jrc_fc2000_ha,
    jrc_perc_DirectDeforestation = (DirectDeforestation / 10000) / jrc_fc2000_ha,
    jrc_perc_DeforAfterDegrad = (DeforAfterDegrad / 10000) / jrc_fc2000_ha,
    jrc_perc_DefTotal = (DirectDeforestation + DeforAfterDegrad) / 10000 / jrc_fc2000_ha
  )

# save data
save(vil_pd, file = "./data/rdat/vil_pd.RData")
