# automation build process
library(dplyr)
library(magrittr)


# Conditions Ids:
# A -> id-1, id-2, id-3, id-4, id-5, id-6, id-7, id-10, (id-42 eeg-quality 0%) => 8 (9) Participents
#condition_list <- c("id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10")
# B -> id-13, id-14, id-15, id-16, id-17b, id-18, id-19, id-20, id-31, id-34, (id-25) 0 => optional participent because subjective it was a none-conscientious
condition_list <- c("id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
# C -> id-21, id-22, id-22, id-23, id-24, id-25, id-26, id-27, id-28, id-29 
#condition_list <- c("id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29")

condition_length <- length(condition_list)
condition <- 'Condition B'

# Source: https://stackoverflow.com/questions/9368900/how-to-check-if-object-variable-is-defined-in-r
existsEnvVariable <-function(name) {
  return(1==length(ls(pattern=paste("^", name, "$", sep=""), env=globalenv())))
}

for (id_ in condition_list) {

  # 0. Clear environment
  source("CleanUpEnvironmentFromTemporaryUsedVariables.r", echo=TRUE)
  
  
  # 1. ImportTrackingData
  source("ImportTrackingData.r", echo=TRUE)
  
  type_vrstk <- 'VRSTK'
  type_biosppy <- 'Biosppy'
  id <- id_
  path <- file.path(condition,  type_vrstk, "/")
  
  #   1.1 ImportTrackingData
  # vrstk tracking files
  #rawTrackingData <- ImportTrackingData('Condition A/VRSTK/Proband-id-1-Condition-A_8-18_15-23-31.json')
  vrstk_files <- list.files(path, pattern=".json", all.files=T, full.names=T)
  for (file in vrstk_files) {
    if(grepl(id, file)){
      rawTrackingData <- ImportTrackingData(file)
      break
    }
  }
  
  #   1.2 ImporttransformedBitalinoTrackingData
  # biosppy feature extracted file
  #transformedBitalinoECGDataFrameStage0 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0-id-1-Condition-A-ECG_HearRateResults.txt')
  #transformedBitalinoEDADataFrameStage0 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0_id-1-Condition-A-EDA_EdaResults.txt')
  
  #transformedBitalinoECGDataFrameStage1 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-ECG_HearRateResults.txt')
  #transformedBitalinoEDADataFrameStage1 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-EDA_EdaResults.txt')
  
  #transformedBitalinoECGDataFrameStage2 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-ECG_HearRateResults.txt')
  #transformedBitalinoEDADataFrameStage2 <- ImporttransformedBitalinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-EDA_EdaResults.txt')
  path <- file.path(condition,  type_biosppy, "/", id, "/")
  biosppy_files <- list.files(path, pattern=".txt", all.files=T, full.names=T)
  for (file in biosppy_files) {
    if(grepl(id, file) && grepl("Stage-0", file) && grepl("ECG_HearRateResults", file)){
      transformedBitalinoECGDataFrameStage0 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoECGDataFrameStage0) <- c('time','HeartRate','RPeaks')
      transformedBitalinoECGDataFrameStage0 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-1", file) && grepl("ECG_HearRateResults", file)){
      transformedBitalinoECGDataFrameStage1 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoECGDataFrameStage1) <- c('time','HeartRate','RPeaks')
      transformedBitalinoECGDataFrameStage1 %<>% mutate_if(is.character, as.numeric)
      transformedBitalinoECGDataFrameStage1$time <- as.integer(transformedBitalinoECGDataFrameStage1$time)
    }
    if(grepl(id, file) && grepl("Stage-2", file) && grepl("ECG_HearRateResults", file)){
      transformedBitalinoECGDataFrameStage2 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoECGDataFrameStage2) <- c('time','HeartRate','RPeaks')
      transformedBitalinoECGDataFrameStage2 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-0", file) && grepl("EDA_EdaResults", file)){
      transformedBitalinoEDADataFrameStage0 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDADataFrameStage0) <- c('onsets','peaks','amps')
      transformedBitalinoEDADataFrameStage0 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-1", file) && grepl("EDA_EdaResults", file)){
      transformedBitalinoEDADataFrameStage1 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDADataFrameStage1) <- c('onsets','peaks','amps')
      transformedBitalinoEDADataFrameStage1 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-2", file) && grepl("EDA_EdaResults", file)){
      transformedBitalinoEDADataFrameStage2 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDADataFrameStage2) <- c('onsets','peaks','amps')
      transformedBitalinoEDADataFrameStage2 %<>% mutate_if(is.character, as.numeric)
    }
  }
  #   1.2.1 Upsampling transformed Bitalino EDA DataFrame
  source("UpsamplingTransformedBitalinoEDADataFrame.R", echo=TRUE)
  transformedBitalinoEDADataFrameStage0 <- upsamplingTransformedBitalinoEdeDataFrame(0)
  transformedBitalinoEDADataFrameStage1 <- upsamplingTransformedBitalinoEdeDataFrame(1)
  transformedBitalinoEDADataFrameStage2 <- upsamplingTransformedBitalinoEdeDataFrame(2)
  
  
  # 2. RawEmotivTrackingData
  source("RawEmotivTrackingData.r", echo=TRUE)
  
  
  # 3. RawBandPowerData
  source("RawBandPowerData.r", echo=TRUE)
  #   3.1 Downsampling/Upsampling
  source("DownsampleToBitalinoResults.r", echo=TRUE) 
  bandPowerDataFrameStage0 <- downsampling(3,0)
  bandPowerDataFrameStage1 <- downsampling(3,1)
  bandPowerDataFrameStage2 <- downsampling(3,2)
  
  
  # 4. RawPerformanceMetricData
  source("RawPerformanceMetricData.r", echo=TRUE)
  #   4.1 Downsampling/Upsampling
  source("DownsampleToBitalinoResults.r", echo=TRUE) 
  performanceMetricDataFrameStage0 <- downsampling(4,0)
  performanceMetricDataFrameStage1 <- downsampling(4,1)
  performanceMetricDataFrameStage2 <- downsampling(4,2)
  
  
  # 5. PagesQualityParameters
  source("PagesQualityParameters.r", echo=TRUE)
  #   5.1 Downsampling/Upsampling
  source("DownsampleToBitalinoResults.r", echo=TRUE) 
  pagesQualityParametersStage1 <- downsampling(5,1)
  pagesQualityParametersStage2 <- downsampling(5,2)
  #   5.2 evaluating Quality parameters
  # pagesTIMESUMsStage1 <- NULL # only activating if it is a new condition group
  
  #lastRow <- nrow(pagesQualityParametersStage1)
  #if (is.null(pagesTIMESUMsStage1)) {
  #  pagesTIMESUMsStage1 <- data.frame("MISSING"  = c(as.numeric(0)), 
  #                                    "TIME_RSI" = c(as.numeric(0)), 
  #                                    "TIME_SUM" = c(as.numeric(pagesQualityParametersStage1[lastRow, 7])), 
  #                                    "MISSREL"  = c(as.numeric(0)), 
  #                                    "DEG_TIME" = c(as.numeric(pagesQualityParametersStage1[lastRow, 10])));
  #} else {
  #  row <- c(as.numeric(0), 
  #           as.numeric(0), 
  #           as.numeric(pagesQualityParametersStage1[lastRow, 7]), 
  #           as.numeric(0), 
  #           as.numeric(pagesQualityParametersStage1[lastRow, 10]))  
    
  #  pagesTIMESUMsStage1Temp <- pagesTIMESUMsStage1                   
  #  pagesTIMESUMsStage1Temp[nrow(pagesTIMESUMsStage1) + 1, ] <- row
  #  pagesTIMESUMsStage1 <- pagesTIMESUMsStage1Temp
  #}
  
  # call EvaluateQualityParamtersAsValidityscore
  
  if (existsEnvVariable("pagesTIMESUMsStage1") && !(is.null(pagesTIMESUMsStage1)))
  {
    lastRow <- nrow(pagesQualityParametersStage1)
    row <- c(as.numeric(0), 
             as.numeric(0), 
             as.numeric(pagesQualityParametersStage1[lastRow, 7]), 
             as.numeric(0), 
             as.numeric(pagesQualityParametersStage1[lastRow, 10]))
    
    pagesTIMESUMsStage1Temp <- pagesTIMESUMsStage1                   
    pagesTIMESUMsStage1Temp[nrow(pagesTIMESUMsStage1) + 1, ] <- row
    pagesTIMESUMsStage1 <- pagesTIMESUMsStage1Temp
  } else
  {
    lastRow <- nrow(pagesQualityParametersStage1)
    pagesTIMESUMsStage1 <- data.frame("MISSING"  = c(as.numeric(0)), 
                                      "TIME_RSI" = c(as.numeric(0)), 
                                      "TIME_SUM" = c(as.numeric(pagesQualityParametersStage1[lastRow, 7])), 
                                      "MISSREL"  = c(as.numeric(0)), 
                                      "DEG_TIME" = c(as.numeric(pagesQualityParametersStage1[lastRow, 10])));
  }
  
  
  
  # 6. RawFixationSaccadsData
  source("RawFixationSaccadesData.r", echo=TRUE)
  #   6.1 Downsampling/Upsampling
  source("DownsampleToBitalinoResults.r", echo=TRUE)
  eyeTrackingInformationStage0 <- downsampling(6,0)
  eyeTrackingInformationStage1 <- downsampling(6,1)
  eyeTrackingInformationStage2 <- downsampling(6,2)
  
  
  # 7. RawVRQuestionnaireToolkitUncannyValleyData
  source("RawVRQuestionnaireToolkitUncannyValleyData.r", echo=TRUE)
  #   7.1 create quality dataframe for all probands to calc. the validityscore
  
  
  # 8. RawVRQuestionnaireToolkitSSQDataFrame
  source("RawVRQuestionnaireToolkitSSQDataFrame.r", echo=TRUE)
  #   8.1 create quality dataframe for all probands to calc. the validityscore
  
  
  # 9. Clear environment
  source("CleanUpEnvironmentFromTemporaryUsedVariables.r", echo=TRUE)
  
  
  #--------------------------------------------------
  # 10. Data-Fusion of one Participant
  #rm(participent_13_DataFrame)
  source("FusionOfTrackingDataOfOneParticipent.r", echo=TRUE)
  participent_variable_name <- paste("participent_", id, "_DataFrame", sep="") #'participent_'+ as.character(id) + '_DataFrame'
  assign(participent_variable_name, fuseParticipentDataFrames(id, condition, 1))
  assign(participent_variable_name, unique(get(participent_variable_name)))
  participent_Log <- duplicated(get(participent_variable_name))
  print(participent_Log)
  assign(participent_variable_name, get(participent_variable_name) %>% distinct(time, .keep_all = TRUE))
  
  # add participent id to dataframe
  nRows <- nrow(get(participent_variable_name))
  #tempDataFrame <- data.frame(matrix(id, nrow = countRows, ncol = 1))
  tempDataFrame <- NULL
  tempDataFrame <- data.frame(pId = character())
  tempDataFrame[1:nRows,] <- id
  tempDataFrame <- cbind(tempDataFrame, get(participent_variable_name))
  assign(participent_variable_name, tempDataFrame)
  
  # cleanup 
  rm(pagesTIMESUMsStage1Temp)
  rm(pagesQualityParametersStage1)
  rm(pagesQualityParametersStage2)
  
  rm(rawVRQuestionnaireToolkitSSQDataFrameStage2)
  rm(rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1)
  
  rm(bandPowerDataFrameStage0)
  rm(bandPowerDataFrameStage1)
  rm(bandPowerDataFrameStage2)
  
  rm(performanceMetricDataFrameStage0)
  rm(performanceMetricDataFrameStage1)
  rm(performanceMetricDataFrameStage2)
  
  rm(eyeTrackingInformationStage0)
  rm(eyeTrackingInformationStage1)
  rm(eyeTrackingInformationStage2)
  
  rm(rawTrackingData)
  rm(tempDataFrame)
  
  rm(transformedBitalinoECGDataFrameStage0)
  rm(transformedBitalinoECGDataFrameStage1)
  rm(transformedBitalinoECGDataFrameStage2)
  
  rm(transformedBitalinoEDADataFrameStage0)
  rm(transformedBitalinoEDADataFrameStage1)
  rm(transformedBitalinoEDADataFrameStage2)
  
  # some misc cleanup
  rm(nRows)
  rm(countRows)
  rm(participent_Log)
  
  #participent_13_DataFrame <- NULL
  #participent_13_DataFrame <- fuseParticipentDataFrames(id, condition, 1)
  #participent_13_DataFrame <- unique(participent_13_DataFrame)
  #participent_13_Log <- duplicated(participent_13_DataFrame)
  #print(participent_13_Log)
  #participent_13_DataFrame <- participent_13_DataFrame %>% distinct(time, .keep_all = TRUE)
}


# temporary fix pagesTIMESUMsStage1[2, 3] = 774.1054
#   10.1 Complete data frame
if (!(is.null(pagesTIMESUMsStage1)) && nrow(pagesTIMESUMsStage1) == condition_length) {
  source("EvaluateQualityParamtersAsValidityscore.r", echo=TRUE)
  pagesTIMESUMsStage1 <- EvaluateTimeRsi()
}

c_index <- 1
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  print(c_index)
  tempDataFrame <- get(dataframe_name)
  
  tempDataFrame$TIMERSICalc   <- pagesTIMESUMsStage1$TIME_RSI[c_index]
  tempDataFrame$MISSRELCalc   <- pagesTIMESUMsStage1$MISSREL[c_index]
  tempDataFrame$MEDIANForTRSI <- pagesTIMESUMsStage1$MEDIANForTRSI[c_index]
  assign(dataframe_name, tempDataFrame)
  
  c_index <- c_index + 1
}


#   10.2 Save data frames as backup
path <- file.path(condition,  "RResults", "/")

pathCSV <- file.path(path,  "PagesTIMESUMsStage1_DataFrame.csv", "")
pathTXT <- file.path(path,  "PagesTIMESUMsStage1_DataFrame.txt", "")
write.csv2(pagesTIMESUMsStage1, pathCSV, row.names = FALSE)
write.table(pagesTIMESUMsStage1, pathTXT, sep=" # ", row.names=FALSE)


for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  file_name_csv <- paste(dataframe_name, ".csv", sep="")
  file_name_txt <- paste(dataframe_name, ".txt", sep="")
  pathCSV <- file.path(path,  file_name_csv, "")
  pathTXT <- file.path(path,  file_name_txt, "")
  
  tempDataFrame <- get(dataframe_name)
  
  write.csv2(tempDataFrame, pathCSV, row.names = FALSE)
  write.table(tempDataFrame, pathTXT, sep=" # ", row.names=FALSE)
}

#pathCSV <- file.path(path,  "Participent_id-1_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-1_DataFrame.txt", "")
#write.csv2(`participent_id-1_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-1_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-2_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-2_DataFrame.txt", "")
#write.csv2(`participent_id-2_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-2_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-3_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-3_DataFrame.txt", "")
#write.csv2(`participent_id-3_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-3_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-4_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-4_DataFrame.txt", "")
#write.csv2(`participent_id-4_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-4_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-5_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-5_DataFrame.txt", "")
#write.csv2(`participent_id-5_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-5_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-6_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-6_DataFrame.txt", "")
#write.csv2(`participent_id-6_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-6_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-7_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-7_DataFrame.txt", "")
#write.csv2(`participent_id-7_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-7_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)
#
#pathCSV <- file.path(path,  "Participent_id-10_DataFrame.csv", "")
#pathTXT <- file.path(path,  "Participent_id-10_DataFrame.txt", "")
#write.csv2(`participent_id-10_DataFrame`, pathCSV, row.names = FALSE)
#write.table(`participent_id-10_DataFrame`, pathTXT, sep=" # ", row.names=FALSE)


#--------------------------------------------------
# 11. Data-Fusion for all Probands
all_participent_dataframe <- NULL
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  
  if (is.null(all_pariticipent_dataframe)){
    all_participent_dataframe <- get(dataframe_name)
  } else {
    all_participent_dataframe <- rbind(all_participent_dataframe, get(dataframe_name))
  }
}

#if (existsEnvVariable("all_pariticipent_dataframe"))
#{
#  all_pariticipent_dataframe <- rbind(all_pariticipent_dataframe, get(participent_variable_name))
#} else
#{
#  all_pariticipent_dataframe <- get(participent_variable_name)
#}

# Condition A DataFrame
#all_participent_dataframe <- `participent_id-1_DataFrame`
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-2_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-3_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-4_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-5_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-6_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-7_DataFrame`)
#all_participent_dataframe <- rbind(all_participent_dataframe, `participent_id-10_DataFrame`)

path <- file.path(condition,  "RResults", "/")

pathCSV <- file.path(path,  "All_Participents_DataFrame.csv", "")
write.csv2(all_participent_dataframe, pathCSV, row.names = FALSE)

pathTXT <- file.path(path,  "All_Participents_DataFrame.txt", "")
write.table(all_participent_dataframe, pathTXT, sep=" # ", row.names=FALSE)

# cleanup globalenv
#rm(get(participent_variable_name))


id <- NULL
condition <- NULL

rm(all_pariticipent_dataframe)
rm(all_participent_dataframe)

rm(pagesTIMESUMsStage1)
rm(tempDataFrame)

for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  remove(list=c(dataframe_name))
}

rm(c_index)
rm(lastRow)
rm(file_name_csv)
rm(dataframe_name)
rm(file_name_txt)
rm(participent_variable_name)
rm(path)
rm(pathCSV)
rm(pathTXT)

#--------------------------------------------------
# 12. Run Cluster-Algorithmen
#   12.1 Create csv-file of last fused Data 
#   12.2 Load it with python as DataFrame
#   12.2 Run and test two selected cluster algorithmen
#     12.2.1 K-Means-Clustering
#     12.2.2 Gaussian-Mixtures-Clustering

#--------------------------------------------------
# 13. Run if there are results ANOVA
# 13.1 Display the covarianz-Matrix of all features
# 13.2 Run ANOVA with covarianz results features
# 13.2 Run ANOVA with the rest features



