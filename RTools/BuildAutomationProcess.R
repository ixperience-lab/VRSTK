# automation build process
library(dplyr)
library(magrittr)
library(psych)
library(stringi)
library(stringr)
library(jsonlite)
library(glue)
library(magrittr)
library(mnormt)


condition <- 'Condition C'
condition_list <- NULL
if (str_detect(condition, "Condition A")){
  # Conditions Ids:
  # A -> id-1, id-2, id-3, id-4, id-5, id-6, id-7, id-10, (id-42 eeg-quality 0%) => 8 (9) Participents
  condition_list <- c("id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10")
} else if (str_detect(condition, "Condition B")){
  # B -> id-13, id-14, id-15, id-16, id-17b, id-18, id-19, id-20, id-31, id-34, (id-25) 0 => optional participent because subjective it was a none-conscientious
  condition_list <- c("id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
  #condition_list <- c("id-13", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
} else if (str_detect(condition, "Condition C")){
  # C -> id-21, id-22, id-22, id-23, id-24, id-25, id-26, id-27, id-28, id-29 
  condition_list <- c("id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29")
}

condition_length <- length(condition_list)

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
  vrstk_files <- list.files(path, pattern=".json", all.files=T, full.names=T)
  for (file in vrstk_files) {
    if(grepl(id, file)){
      rawTrackingData <- ImportTrackingData(file)
      break
    }
  }
  
  #   1.2 ImporttransformedBitalinoTrackingData
  # biosppy feature extracted file
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
    if(grepl(id, file) && grepl("Stage-0", file) && grepl("EDA_SiemensEdaResults", file)){
      transformedBitalinoEDASiemensDataFrameStage0 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDASiemensDataFrameStage0) <- c('RawValueInMicroSiemens','FilteredValueInMicroSiemens')
      transformedBitalinoEDASiemensDataFrameStage0 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-1", file) && grepl("EDA_SiemensEdaResults", file)){
      transformedBitalinoEDASiemensDataFrameStage1 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDASiemensDataFrameStage1) <- c('RawValueInMicroSiemens','FilteredValueInMicroSiemens')
      transformedBitalinoEDASiemensDataFrameStage1 %<>% mutate_if(is.character, as.numeric)
    }
    if(grepl(id, file) && grepl("Stage-2", file) && grepl("EDA_SiemensEdaResults", file)){
      transformedBitalinoEDASiemensDataFrameStage2 <- ImportTransformedBitalinoTrackingData(file)
      colnames(transformedBitalinoEDASiemensDataFrameStage2) <- c('RawValueInMicroSiemens','FilteredValueInMicroSiemens')
      transformedBitalinoEDASiemensDataFrameStage2 %<>% mutate_if(is.character, as.numeric)
    }
  }
  #   1.2.1 Upsampling transformed Bitalino EDA DataFrame
  source("UpsamplingTransformedBitalinoEDADataFrame.R", echo=TRUE)
  transformedBitalinoEDADataFrameStage0 <- upsamplingTransformedBitalinoEdeDataFrame(0)
  transformedBitalinoEDADataFrameStage1 <- upsamplingTransformedBitalinoEdeDataFrame(1)
  transformedBitalinoEDADataFrameStage2 <- upsamplingTransformedBitalinoEdeDataFrame(2)
  
  
  #   1.2.2 Downsampling Bitalino EDA Siemens DataFrame
  source("DownsampleToBitalinoResults.r", echo=TRUE) 
  transformedBitalinoEDASiemensDataFrameStage0 <- downsampling(2,0)
  transformedBitalinoEDASiemensDataFrameStage1 <- downsampling(2,1)
  transformedBitalinoEDASiemensDataFrameStage2 <- downsampling(2,2)
  
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
  source("FusionOfTrackingDataOfOneParticipent.r", echo=TRUE)
  participent_variable_name <- paste("participent_", id, "_DataFrame", sep="") #'participent_'+ as.character(id) + '_DataFrame'
  # call fuseParticipentDataFrames
  assign(participent_variable_name, fuseParticipentDataFrames(id, condition, 1))
  # filter duplicates
  assign(participent_variable_name, unique(get(participent_variable_name)))
  # print filter log
  participent_Log <- duplicated(get(participent_variable_name))
  print(participent_Log)
  
  assign(participent_variable_name, get(participent_variable_name) %>% distinct(time, .keep_all = TRUE))
  
  # 10.1 add participent id to dataframe
  nRows <- nrow(get(participent_variable_name))
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
  
}


#   10.2 Complete data frame
if (!(is.null(pagesTIMESUMsStage1)) && nrow(pagesTIMESUMsStage1) == condition_length) {
  source("EvaluateQualityParamtersAsValidityscore.r", echo=TRUE)
  pagesTIMESUMsStage1 <- EvaluateTimeRsi()
}

c_index <- 1
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  
  tempDataFrame <- get(dataframe_name)
  tempDataFrame$TIMERSICalc   <- pagesTIMESUMsStage1$TIME_RSI[c_index]
  tempDataFrame$MISSRELCalc   <- pagesTIMESUMsStage1$MISSREL[c_index]
  tempDataFrame$MEDIANForTRSI <- pagesTIMESUMsStage1$MEDIANForTRSI[c_index]
  assign(dataframe_name, tempDataFrame)
  
  c_index <- c_index + 1
}


#   10.3 Filter data frame and complete with missing values
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  
  tempDataFrame <- get(dataframe_name)
  
  # delete "NA" rows 
  tempDataFrame <- na.omit(tempDataFrame)
  
  # convert from bool to numeric
  numberOfRows <- nrow(tempDataFrame)
  if ( tempDataFrame$DEG_TIME[numberOfRows] > 0 && tempDataFrame$DEG_TIME[numberOfRows] >= tempDataFrame$DegTimeThreshold[numberOfRows]){
    tempDataFrame$DegTimeLowQuality = 1
  }else {
    tempDataFrame$DegTimeLowQuality = 0
  }
  tempDataFrame$DegTimeLowQuality <- as.numeric(tempDataFrame$DegTimeLowQuality)
  
  # remove known columns with zeros
  tempDataFrame$MISSING     <- NULL
  tempDataFrame$TIME_RSI    <- NULL
  tempDataFrame$MISSRELCalc <- NULL
  
  # calculate tatal features again, while there is a bug
  # TotalFixationCounter
  tempDataFrame$TotalFixationCounter[1] <- tempDataFrame$FixationCounter[1]
  # TotalFixationDuration
  tempDataFrame$TotalFixationDuration[1] <- tempDataFrame$FixationDuration[1]
  # SaccadeCounter
  tempColumn <- tempDataFrame$SaccadeCounter
  tempDataFrame$SaccadeCounter <- 0
  #tempDataFrame$SaccadeCounter[1] <- 0
  for(i in 2:nrow(tempDataFrame)) {
    tempDataFrame$TotalFixationCounter[i] <- tempDataFrame$TotalFixationCounter[i-1] + tempDataFrame$FixationCounter[i]
    tempDataFrame$TotalFixationDuration[i] <- tempDataFrame$TotalFixationDuration[i-1] + tempDataFrame$FixationDuration[i]
    tempDataFrame$SaccadeCounter[i] <- tempDataFrame$SaccadeCounter[i-1] + (tempColumn[i] - tempColumn[i-1])
  }
  
  # pupilometry change of pupil diameter over time in seconds
  # LeftEyeOpenness, RightEyeOpenness
  # LeftPupilDiameter, RightPupilDiameter
  #
  # Book: Advances in Artificial Intelligence and Applied Cognitive Computing
  # Subchapter: 2.1 TEPR, p. 1057
  # Content: It was found that changes in pupil diameter size evoked by light reflexes can be described as large (up to a few millimeters), while those evoked by
  # cognitive activity happen to be relatively small (usually between 0.1 and 0.5 mm) and rapid [1].
  leftMeanPupilDiameter <- mean(tempDataFrame$LeftPupilDiameter)
  rightMeanPupilDiameter <- mean(tempDataFrame$RightPupilDiameter)
  
  tempDataFrame$LightReflexesLeftPupilDiamter     <- 0
  tempDataFrame$CognitiveActivityLeftPupilDiamter <- 0
  tempDataFrame$LeftMeanPupilDiameter             <- leftMeanPupilDiameter
  tempDataFrame$LeftPupilDiameterDifferenceToMean[1] <- tempDataFrame$LeftPupilDiameter[1] - leftMeanPupilDiameter
  
  tempDataFrame$LightReflexesRightPupilDiamter        <- 0
  tempDataFrame$CognitiveActivityRightPupilDiamter    <- 0
  tempDataFrame$RightMeanPupilDiameter                <- rightMeanPupilDiameter
  tempDataFrame$RightPupilDiameterDifferenceToMean[1] <- tempDataFrame$RightPupilDiameter[1] - rightMeanPupilDiameter
  
  for(i in 2:nrow(tempDataFrame)) {
    leftDifference <- abs(tempDataFrame$LeftPupilDiameter[i-1] - tempDataFrame$LeftPupilDiameter[i])
    if(leftDifference > 0.5){
      tempDataFrame$LightReflexesLeftPupilDiamter[i] <- leftDifference
    } else {
      tempDataFrame$CognitiveActivityLeftPupilDiamter[i] <- leftDifference
    }
    
    rightDifference <- abs(tempDataFrame$RightPupilDiameter[i-1] - tempDataFrame$RightPupilDiameter[i])
    if(rightDifference > 0.5){
      tempDataFrame$LightReflexesRightPupilDiamter[i] <- rightDifference
    } else {
      tempDataFrame$CognitiveActivityRightPupilDiamter[i] <- rightDifference
    }
    
    tempDataFrame$LeftPupilDiameterDifferenceToMean[i]  <- tempDataFrame$LeftPupilDiameter[i]  - leftMeanPupilDiameter
    tempDataFrame$RightPupilDiameterDifferenceToMean[i] <- tempDataFrame$RightPupilDiameter[i] - rightMeanPupilDiameter
  }
  
  assign(dataframe_name, tempDataFrame)
}


#   10.4 Save data frames as backup
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



#--------------------------------------------------
# 11. Data-Fusion for all Probands

all_participent_dataframe <- NULL
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="") 
  
  if (is.null(all_participent_dataframe)){
    all_participent_dataframe <- get(dataframe_name)
  } else {
    all_participent_dataframe <- rbind(all_participent_dataframe, get(dataframe_name))
  }
}

path <- file.path(condition,  "RResults", "/")

pathCSV <- file.path(path,  "All_Participents_DataFrame.csv", "")
write.csv2(all_participent_dataframe, pathCSV, row.names = FALSE)

pathTXT <- file.path(path,  "All_Participents_DataFrame.txt", "")
write.table(all_participent_dataframe, pathTXT, sep=" # ", row.names=FALSE)

id <- NULL
condition <- NULL

rm(all_pariticipent_dataframe)
rm(all_participent_dataframe)

rm(transformedBitalinoEDASiemensDataFrameStage0)
rm(transformedBitalinoEDASiemensDataFrameStage1)
rm(transformedBitalinoEDASiemensDataFrameStage2)

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
rm(i)
rm(id)
rm(id_)
rm(leftDifference)
rm(leftMeanPupilDiameter)
rm(numberOfRows)
rm(rightDifference)
rm(rightMeanPupilDiameter)
rm(tempColumn)
rm(cid)

# -------------------------------------- End of one Condition fusion -----------------------------------


# 12 Fuse all conditions dataframes
conditionADataFrame <- read.csv2(file = "./Condition A/RResults/All_Participents_DataFrame.csv")
#head(conditionADataFrame)

conditionBDataFrame <- read.csv2(file = "./Condition B/RResults/All_Participents_DataFrame.csv")
#head(conditionBDataFrame)

conditionCDataFrame <- read.csv2(file = "./Condition C/RResults/All_Participents_DataFrame.csv")
#head(conditionBDataFrame)


all_participent_dataframe <- conditionADataFrame
all_participent_dataframe <- rbind(all_participent_dataframe, conditionBDataFrame)
all_participent_dataframe <- rbind(all_participent_dataframe, conditionCDataFrame)

temp_dataframe <- all_participent_dataframe

# all cells with "NA" = 0 
#temp_dataframe <- replace(temp_dataframe, is.na(temp_dataframe), 0)
# all rows with "NA" to cut out
temp_dataframe <- na.omit(temp_dataframe)

# clean ids -> only numbers
temp_dataframe$pId <- str_replace(temp_dataframe$pId, "id-", "")
temp_dataframe$pId <- str_replace(temp_dataframe$pId, "b", "")

# switch time and id
numberOfColumns <- ncol(temp_dataframe)
temp_dataframe <- temp_dataframe[, c(2,1,3:numberOfColumns)] 

# remove strings
temp_dataframe <- subset(temp_dataframe, select=-c(lex, STARTED, LASTDATA, MAXPAGE, DegTimeThreshold, DegTimeThresholdForOnePage, DegTimeValueForOnePage)) # MAXPAGE is optional

# convert all columns to numeric
numberOfColumns <- ncol(temp_dataframe)
temp_dataframe[,1:numberOfColumns] <- lapply(temp_dataframe[,1:numberOfColumns], function (x) as.numeric(x))

# list <- sapply(temp_dataframe, class)
write.csv2(temp_dataframe, "./All_Participents_DataFrame.csv", row.names = FALSE)

rm(all_participent_dataframe)
rm(conditionADataFrame)
rm(conditionBDataFrame)
rm(conditionCDataFrame)
rm(temp_dataframe)
rm(list)


# 12.1 create fuse mean data frame all participants

condition <- 'Condition A'
condition_list <- NULL
if (str_detect(condition, "Condition A")){
  # Conditions Ids:
  # A -> id-1, id-2, id-3, id-4, id-5, id-6, id-7, id-10, (id-42 eeg-quality 0%) => 8 (9) Participents
  condition_list <- c("id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10")
} else if (str_detect(condition, "Condition B")){
  # B -> id-13, id-14, id-15, id-16, id-17b, id-18, id-19, id-20, id-31, id-34, (id-25) 0 => optional participent because subjective it was a none-conscientious
  condition_list <- c("id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
  #condition_list <- c("id-13", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
} else if (str_detect(condition, "Condition C")){
  # C -> id-21, id-22, id-22, id-23, id-24, id-25, id-26, id-27, id-28, id-29 
  condition_list <- c("id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29")
}

condition_length <- length(condition_list)

# Conditions Ids:
# A -> id-1, id-2, id-3, id-4, id-5, id-6, id-7, id-10, (id-42 eeg-quality 0%) => 8 (9) Participents
#condition_list <- c("id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10")
# B -> id-13, id-14, id-15, id-16, id-17b, id-18, id-19, id-20, id-31, id-34, (id-25) 0 => optional participent because subjective it was a none-conscientious
#condition_list <- c("id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
#condition_list <- c("id-13", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
# C -> id-21, id-22, id-22, id-23, id-24, id-25, id-26, id-27, id-28, id-29 
#condition_list <- c("id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29")

#condition <- 'Condition B'
path <- file.path(condition,  "RResults", "/")

All_Participents_Mean_DataFrame <- NULL
for (cid in condition_list){ 
  dataframe_name <- paste("participent_", cid, "_DataFrame", sep="")
  file_name <- paste (dataframe_name, ".csv", sep="")
  
  pathCSV <- file.path(path,  file_name, "")
  temp_dataframe <- read.csv2(file = pathCSV)
  
  # alls cells with "NA" = 0
  #temp_dataframe <- replace(temp_dataframe, is.na(temp_dataframe), 0)
  # all rows with "NA" to cut out
  temp_dataframe <- na.omit(temp_dataframe)
  
  # clean ids -> only numbers
  temp_dataframe$pId <- str_replace(temp_dataframe$pId, "id-", "")
  temp_dataframe$pId <- str_replace(temp_dataframe$pId, "b", "")
  
  # switch time and id
  numberOfColumns <- ncol(temp_dataframe)
  temp_dataframe <- temp_dataframe[, c(2,1,3:numberOfColumns)] # leave the row index blank to keep all rows
  
  # remove strings
  temp_dataframe <- subset(temp_dataframe, select=-c(lex, STARTED, LASTDATA, MAXPAGE, DegTimeThreshold, DegTimeThresholdForOnePage, DegTimeValueForOnePage)) # MAXPAGE is optional
  
  # convert all columns to numeric
  numberOfColumns <- ncol(temp_dataframe)
  temp_dataframe[,1:numberOfColumns] <- lapply(temp_dataframe[,1:numberOfColumns], function (x) as.numeric(x))
  
  meanArray <- colMeans(temp_dataframe)  
  temp_df <- data.frame(matrix(0, ncol = length(meanArray), nrow = 0))
  temp_df <- rbind(temp_df, meanArray)
  colnames(temp_df) <- colnames(temp_dataframe)
  
  if(is.null(All_Participents_Mean_DataFrame)){
    All_Participents_Mean_DataFrame <- temp_df
  }else{
    All_Participents_Mean_DataFrame <- rbind(All_Participents_Mean_DataFrame, temp_df)
  }
}

path <- file.path(condition,  "RResults", "/")
path_file <- file.path(path,  "All_Participents_Mean_DataFrame.csv", "")
write.csv2(All_Participents_Mean_DataFrame, path_file, row.names = FALSE)


# 12.4.2 create complete fuse mean data frame all participants
conditionAMeanDataFrame <- read.csv2(file = "./Condition A/RResults/All_Participents_Mean_DataFrame.csv")
#head(conditionAMeanDataFrame)

conditionBMeanDataFrame <- read.csv2(file = "./Condition B/RResults/All_Participents_Mean_DataFrame.csv")
#head(conditionBMeanDataFrame)

conditionCMeanDataFrame <- read.csv2(file = "./Condition C/RResults/All_Participents_Mean_DataFrame.csv")
#head(conditionBMeanDataFrame)

all_participent_Mean_dataframes <- conditionAMeanDataFrame
all_participent_Mean_dataframes <- rbind(all_participent_Mean_dataframes, conditionBMeanDataFrame)
all_participent_Mean_dataframes <- rbind(all_participent_Mean_dataframes, conditionCMeanDataFrame)

write.csv2(all_participent_Mean_dataframes, "All_Participents_Mean_DataFrame.csv", row.names = FALSE)


#--------------------------------------------------
# 13. Run Cluster-Algorithmen
# 13.1 Create csv-file of last fused Data 
# 13.2 Load it with python as DataFrame
# 13.2 Run and test two selected cluster algorithmen
# 13.2.1 K-Means-Clustering
# 13.2.2 Gaussian-Mixtures-Clustering

#--------------------------------------------------
# 14. Run if there are results ANOVA
# 14.1 Display the covarianz-Matrix of all features
# 14.2 Run ANOVA with covarianz results features
# 14.2 Run ANOVA with the rest features



