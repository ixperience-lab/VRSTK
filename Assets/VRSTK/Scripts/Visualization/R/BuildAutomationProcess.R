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


for (condition in c("Condition A", "Condition B", "Condition C")){
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
      # ECG_HearRateResults
      if(grepl(id, file) && grepl("Stage-0", file) && grepl("ECG_HearRateResults", file)){
        transformedBitalinoECGDataFrameStage0 <- ImportTransformedBitalinoTrackingData(file)
        colnames(transformedBitalinoECGDataFrameStage0) <- c('time','HeartRate','RPeaks')
        transformedBitalinoECGDataFrameStage0 %<>% mutate_if(is.character, as.numeric)
        transformedBitalinoECGDataFrameStage0$time <- as.integer(transformedBitalinoECGDataFrameStage0$time)
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
        transformedBitalinoECGDataFrameStage2$time <- as.integer(transformedBitalinoECGDataFrameStage2$time)
      }
      # ECG_HearRateVariabilityResults
      if(grepl(id, file) && grepl("Stage-0", file) && grepl("ECG_HearRateVariabilityResults", file)){
        transformedBitalinoECGHRVDataFrameStage0 <- ImportTransformedBitalinoTrackingData(file)
        colnames(transformedBitalinoECGHRVDataFrameStage0) <- c('RRI', 'RRMin', 'RRMean', 'RRMax', 'SDSD', 'SD1', 'SD2', 'SDNN', 'RMSSD', 'SD1SD2Ratio',
                                                                'SEllipseArea', 'VLFPeak', 'LFPeak', 'HFPeak', 'VLFAbs', 'LFAbs', 'HFAbs', 'VLFLog', 'LFLog', 'HFLog',
                                                                'LFNorm', 'HFNorm', 'LFHFRatio', 'FBTotal')
        transformedBitalinoECGHRVDataFrameStage0 %<>% mutate_if(is.character, as.numeric)
      }
      if(grepl(id, file) && grepl("Stage-1", file) && grepl("ECG_HearRateVariabilityResults", file)){
        transformedBitalinoECGHRVDataFrameStage1 <- ImportTransformedBitalinoTrackingData(file)
        colnames(transformedBitalinoECGHRVDataFrameStage1) <- c('RRI', 'RRMin', 'RRMean', 'RRMax', 'SDSD', 'SD1', 'SD2', 'SDNN', 'RMSSD', 'SD1SD2Ratio',
                                                                'SEllipseArea', 'VLFPeak', 'LFPeak', 'HFPeak', 'VLFAbs', 'LFAbs', 'HFAbs', 'VLFLog', 'LFLog', 'HFLog',
                                                                'LFNorm', 'HFNorm', 'LFHFRatio', 'FBTotal')
        transformedBitalinoECGHRVDataFrameStage1 %<>% mutate_if(is.character, as.numeric)
      }
      if(grepl(id, file) && grepl("Stage-2", file) && grepl("ECG_HearRateVariabilityResults", file)){
        transformedBitalinoECGHRVDataFrameStage2 <- ImportTransformedBitalinoTrackingData(file)
        colnames(transformedBitalinoECGHRVDataFrameStage2) <- c('RRI', 'RRMin', 'RRMean', 'RRMax', 'SDSD', 'SD1', 'SD2', 'SDNN', 'RMSSD', 'SD1SD2Ratio',
                                                                'SEllipseArea', 'VLFPeak', 'LFPeak', 'HFPeak', 'VLFAbs', 'LFAbs', 'HFAbs', 'VLFLog', 'LFLog', 'HFLog',
                                                                'LFNorm', 'HFNorm', 'LFHFRatio', 'FBTotal')
        transformedBitalinoECGHRVDataFrameStage2 %<>% mutate_if(is.character, as.numeric)
      }
      # EDA_EdaResults
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
      # EDA_SiemensEdaResults
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
    
    #   1.2.3 Add time from ecg dataframe to ecg hrv dataframe
    transformedBitalinoECGHRVDataFrameStage0$time <- transformedBitalinoECGDataFrameStage0$time
    transformedBitalinoECGHRVDataFrameStage1$time <- transformedBitalinoECGDataFrameStage1$time
    transformedBitalinoECGHRVDataFrameStage2$time <- transformedBitalinoECGDataFrameStage2$time
    
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
    # stage 1
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
    # stage 2
    if (existsEnvVariable("pagesTIMESUMsStage2") && !(is.null(pagesTIMESUMsStage2)))
    {
      lastRow <- nrow(pagesQualityParametersStage2)
      row <- c(as.numeric(0), 
               as.numeric(0), 
               as.numeric(pagesQualityParametersStage2[lastRow, 7]), 
               as.numeric(0), 
               as.numeric(pagesQualityParametersStage2[lastRow, 10]))
      
      pagesTIMESUMsStage2Temp <- pagesTIMESUMsStage2                   
      pagesTIMESUMsStage2Temp[nrow(pagesTIMESUMsStage2) + 1, ] <- row
      pagesTIMESUMsStage2 <- pagesTIMESUMsStage2Temp
    } else
    {
      lastRow <- nrow(pagesQualityParametersStage2)
      pagesTIMESUMsStage2 <- data.frame("MISSING"  = c(as.numeric(0)), 
                                        "TIME_RSI" = c(as.numeric(0)), 
                                        "TIME_SUM" = c(as.numeric(pagesQualityParametersStage2[lastRow, 7])), 
                                        "MISSREL"  = c(as.numeric(0)), 
                                        "DEG_TIME" = c(as.numeric(pagesQualityParametersStage2[lastRow, 10])));
    }
    
    
    # 6. RawFixationSaccadsData
    source("RawFixationSaccadesData.r", echo=TRUE)
    #   6.1 Downsampling/Upsampling
    source("DownsampleToBitalinoResults.r", echo=TRUE)
    eyeTrackingInformationStage0 <- downsampling(6,0)
    eyeTrackingInformationStage1 <- downsampling(6,1)
    eyeTrackingInformationStage2 <- downsampling(6,2)
    

    # 7. CenterOfViewData only for stage 1 data frame
    source("CenterOfViewData.r", echo=TRUE)
    #   7.1 Downsampling/Upsampling
    centerOfViewInformationDataFrameStage1 <- downsampling(7,1)

    
    # 8. RawFixationSaccadsData
    source("RawFixationSaccadesPositionsData.r", echo=TRUE)
    #   8.1 Downsampling/Upsampling
    source("DownsampleToBitalinoResults.r", echo=TRUE)
    eyeTrackingSaccadesInformationStage0 <- downsampling(8,0)
    eyeTrackingSaccadesInformationStage1 <- downsampling(8,1)    
    eyeTrackingSaccadesInformationStage2 <- downsampling(8,2)
    
    # 9. RawVRQuestionnaireToolkitUncannyValleyData
    source("RawVRQuestionnaireToolkitUncannyValleyData.r", echo=TRUE)
    #   9.1 create quality dataframe for all probands to calc. the validityscore
    
    
    # 10. RawVRQuestionnaireToolkitSSQDataFrame
    source("RawVRQuestionnaireToolkitSSQDataFrame.r", echo=TRUE)
    #   10.1 create quality dataframe for all probands to calc. the validityscore
    
    
    # 11. Clear environment
    source("CleanUpEnvironmentFromTemporaryUsedVariables.r", echo=TRUE)
    
    
    #--------------------------------------------------
    # 12. Data-Fusion of one Participant 
    source("FusionOfTrackingDataOfOneParticipent.r", echo=TRUE)
    
    # 12.1.1 Data-Fusion of one Participant stage 0
    participent_variable_name <- paste("participent_", id, "_Stage0_DataFrame", sep="") #'participent_'+ as.character(id) + '_DataFrame'
    # call fuseParticipentDataFrames
    assign(participent_variable_name, fuseParticipentDataFrames(id, condition, 0))
    # filter duplicates
    assign(participent_variable_name, unique(get(participent_variable_name)))
    # print filter log
    participent_Log <- duplicated(get(participent_variable_name))
    print(participent_Log)
    
    assign(participent_variable_name, get(participent_variable_name) %>% distinct(time, .keep_all = TRUE))
    
    # 12.1.2.1 add participent id to dataframe
    nRows <- nrow(get(participent_variable_name))
    tempDataFrame <- NULL
    tempDataFrame <- data.frame(pId = character())
    tempDataFrame[1:nRows,] <- id
    tempDataFrame <- cbind(tempDataFrame, get(participent_variable_name))
    assign(participent_variable_name, tempDataFrame)
    
    # 12.1.2 Data-Fusion of one Participant stage 1
    participent_variable_name <- paste("participent_", id, "_Stage1_DataFrame", sep="") #'participent_'+ as.character(id) + '_DataFrame'
    # call fuseParticipentDataFrames
    assign(participent_variable_name, fuseParticipentDataFrames(id, condition, 1))
    # filter duplicates
    assign(participent_variable_name, unique(get(participent_variable_name)))
    # print filter log
    participent_Log <- duplicated(get(participent_variable_name))
    print(participent_Log)
    
    assign(participent_variable_name, get(participent_variable_name) %>% distinct(time, .keep_all = TRUE))
    
    # 12.1.2.1 add participent id to dataframe
    nRows <- nrow(get(participent_variable_name))
    tempDataFrame <- NULL
    tempDataFrame <- data.frame(pId = character())
    tempDataFrame[1:nRows,] <- id
    tempDataFrame <- cbind(tempDataFrame, get(participent_variable_name))
    assign(participent_variable_name, tempDataFrame)
    
    # 12.1.3 Data-Fusion of one Participant stage 2
    participent_variable_name <- paste("participent_", id, "_Stage2_DataFrame", sep="") #'participent_'+ as.character(id) + '_DataFrame'
    # call fuseParticipentDataFrames
    assign(participent_variable_name, fuseParticipentDataFrames(id, condition, 2))
    # filter duplicates
    assign(participent_variable_name, unique(get(participent_variable_name)))
    # print filter log
    participent_Log <- duplicated(get(participent_variable_name))
    print(participent_Log)
    
    assign(participent_variable_name, get(participent_variable_name) %>% distinct(time, .keep_all = TRUE))
    
    # 12.1.4.1 add participent id to dataframe
    nRows <- nrow(get(participent_variable_name))
    tempDataFrame <- NULL
    tempDataFrame <- data.frame(pId = character())
    tempDataFrame[1:nRows,] <- id
    tempDataFrame <- cbind(tempDataFrame, get(participent_variable_name))
    assign(participent_variable_name, tempDataFrame)
    
    # 12.2 cleanup 
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
  
  
  #   12.3 Complete dataframes
  if (!(is.null(pagesTIMESUMsStage1)) && nrow(pagesTIMESUMsStage1) == condition_length) {
    source("EvaluateQualityParamtersAsValidityscore.r", echo=TRUE)
    pagesTIMESUMsStage1 <- EvaluateValidityScores(1)
  }
  if (!(is.null(pagesTIMESUMsStage2)) && nrow(pagesTIMESUMsStage2) == condition_length) {
    source("EvaluateQualityParamtersAsValidityscore.r", echo=TRUE)
    pagesTIMESUMsStage2 <- EvaluateValidityScores(2)
  }
  
  c_index <- 1
  for (cid in condition_list){ 
    for(stage in c("Stage1", "Stage2")){
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      pagesTIMESUMs_name <- paste("pagesTIMESUMs", stage, sep="") 
      tempDataFrame <- get(dataframe_name)
      pagesTIMESUMs <- get(pagesTIMESUMs_name)
      tempDataFrame$TIMERSICalc          <- pagesTIMESUMs$TIME_RSI[c_index]
      tempDataFrame$MISSRELCalc          <- pagesTIMESUMs$MISSREL[c_index]
      tempDataFrame$MEDIANForTRSI        <- pagesTIMESUMs$MEDIANForTRSI[c_index]
      tempDataFrame$EvaluatedTIMERSICalc <- pagesTIMESUMs$EvaluatedTIMERSICalc[c_index]
      assign(dataframe_name, tempDataFrame)
    }
    c_index <- c_index + 1
  }

  
  #   12.4 Filter data frame and complete with missing values
  for (cid in condition_list){ 
    for(stage in c("Stage0", "Stage1", "Stage2")){
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      
      tempDataFrame <- get(dataframe_name)
      
      # delete "NA" rows 
      tempDataFrame <- na.omit(tempDataFrame)
      
      # convert from bool to numeric
      numberOfRows <- nrow(tempDataFrame)
      
      if(stage != "Stage0"){
      
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
      }
      
      # calculate tatal features again, while there is a bug
      # TotalFixationCounter
      tempDataFrame$TotalFixationCounter[1] <- tempDataFrame$FixationCounter[1]
      # TotalFixationDuration
      tempDataFrame$TotalFixationDuration[1] <- tempDataFrame$FixationDuration[1]
      # SaccadeCounter
      tempColumn <- tempDataFrame$SaccadeCounter
      tempDataFrame$SaccadeCounter <- 0
      #tempDataFrame$SaccadeCounter[1] <- 0
      numberOfRows <- nrow(tempDataFrame)
      if (numberOfRows > 1){
        for(i in 2:nrow(tempDataFrame)) {
          tempDataFrame$TotalFixationCounter[i]  <- tempDataFrame$TotalFixationCounter[i-1]  + tempDataFrame$FixationCounter[i]
          tempDataFrame$TotalFixationDuration[i] <- tempDataFrame$TotalFixationDuration[i-1] + tempDataFrame$FixationDuration[i]
          tempDataFrame$SaccadeCounter[i]        <- tempDataFrame$SaccadeCounter[i-1]        + (tempColumn[i] - tempColumn[i-1])
        }
      } else {
        tempDataFrame$SaccadeCounter[1] <- tempColumn[1]
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
      
      if(stage != "Stage1"){
        # calculate saccads informations
        for(i in 1:nrow(tempDataFrame)) {
          tempDataFrame$SaccadesDiffX[i] <- abs(tempDataFrame$Saccade0X[i] - tempDataFrame$Saccade1X[i])
        }
      }
      
      if(stage == "Stage1"){
        # calculate saccads informations
        # Q-1-ROI (h=20px; w=560; scale_factor=0.005): X[1.465, -1.37]; Y[1.73, 1.83]; Z[-7]
        # Q-2-ROI (h=20px; w=560; scale_factor=0.005): X[1.465, -1.37]; Y[1.27, 1.37]; Z[-7]
        # Q-3-ROI (h=20px; w=560; scale_factor=0.005): X[1.465, -1.37]; Y[0.8, 0.9];   Z[-7]
        q1XSaccads   <- array(numeric())
        indexQ1      <- array(numeric())
        q2XSaccads   <- array(numeric())
        indexQ2      <- array(numeric())
        q3XSaccads   <- array(numeric())
        indexQ3      <- array(numeric())
        oldPageIndex <- 1
        
        for(i in 1:nrow(tempDataFrame)) {
          tempDataFrame$SaccadesDiffX[i] <- abs(tempDataFrame$Saccade0X[i] - tempDataFrame$Saccade1X[i])
          if(tempDataFrame$CurrentPageNumbe[i] > 0){
            if(tempDataFrame$CurrentPageNumbe[i] != oldPageIndex){
              
              if (length(indexQ1) > 0 ){
                tempDataFrame$SaccadesMeanX[indexQ1] <- mean(q1XSaccads)
                #print("saccades info:")
                #print(tempDataFrame$SaccadesMeanX[indexQ1])
                if(length(indexQ1) > 1) {
                  tempDataFrame$SaccadesSdX[indexQ1]   <- sd(q1XSaccads)
                }
                tempDataFrame$SaccadesMinX[indexQ1]  <- min(q1XSaccads)
                tempDataFrame$SaccadesMaxX[indexQ1]  <- max(q1XSaccads)
                q1XSaccads <- array(numeric())
                indexQ1    <- array(numeric())
              }
              
              if (length(indexQ2) > 0 ){
                tempDataFrame$SaccadesMeanX[indexQ2] <- mean(q2XSaccads)
                if(length(indexQ2) > 1) {
                  tempDataFrame$SaccadesSdX[indexQ2]   <- sd(q2XSaccads)
                }
                tempDataFrame$SaccadesMinX[indexQ2]  <- min(q2XSaccads)
                tempDataFrame$SaccadesMaxX[indexQ2]  <- max(q2XSaccads)
                q2XSaccads <- array(numeric())
                indexQ2    <- array(numeric())
              }
              
              if (length(indexQ3) > 0 ){
                tempDataFrame$SaccadesMeanX[indexQ3] <- mean(q3XSaccads)
                if(length(indexQ3) > 1) {
                  tempDataFrame$SaccadesSdX[indexQ3]   <- sd(q3XSaccads)
                }
                tempDataFrame$SaccadesMinX[indexQ3]  <- min(q3XSaccads)
                tempDataFrame$SaccadesMaxX[indexQ3]  <- max(q3XSaccads)
                q3XSaccads <- array(numeric())
                indexQ3    <- array(numeric())
              }
            }
            
            # Q-1-ROI:
            if((tempDataFrame$Saccade0X[i] >= -1.4 && tempDataFrame$Saccade0X[i] <=  1.48) &&
               (tempDataFrame$Saccade0Y[i] >= 1.67  && tempDataFrame$Saccade0Y[i] <=  1.91)   &&
               (tempDataFrame$Saccade0Z[i] < -6.9) &&
               (tempDataFrame$Saccade1X[i] >= -1.4 && tempDataFrame$Saccade1X[i] <=  1.48) &&
               (tempDataFrame$Saccade1Y[i] >= 1.67  && tempDataFrame$Saccade1Y[i] <=  1.91)   &&
               (tempDataFrame$Saccade1Z[i] < -6.9)) {
              tempDataFrame$QuestionId[i] <- 1
              q1XSaccads <- append(q1XSaccads, tempDataFrame$SaccadesDiffX[i])
              indexQ1    <- append(indexQ1, i)
            }
            
            # Q-2-ROI:
            if((tempDataFrame$Saccade0X[i] >= -1.4 && tempDataFrame$Saccade0X[i] <=  1.48) &&
               (tempDataFrame$Saccade0Y[i] >= 1.22  && tempDataFrame$Saccade0Y[i] <=  1.42)  &&
               (tempDataFrame$Saccade0Z[i] < -6.9) &&
               (tempDataFrame$Saccade1X[i] >= -1.4 && tempDataFrame$Saccade1X[i] <=  1.48) &&
               (tempDataFrame$Saccade1Y[i] >= 1.22  && tempDataFrame$Saccade1Y[i] <=  1.42)  &&
               (tempDataFrame$Saccade1Z[i] < -6.9)) {
              tempDataFrame$QuestionId[i] <- 2
              q2XSaccads <- append(q2XSaccads, tempDataFrame$SaccadesDiffX[i])
              indexQ2    <- append(indexQ2, i)
            }
            
            # Q-3-ROI:
            if((tempDataFrame$Saccade0X[i] >= -1.4 && tempDataFrame$Saccade0X[i] <=  1.48) &&
               (tempDataFrame$Saccade0Y[i] >= 0.75  && tempDataFrame$Saccade0Y[i] <=  0.95)  &&
               (tempDataFrame$Saccade0Z[i] < -6.9) &&
               (tempDataFrame$Saccade1X[i] >= -1.4 && tempDataFrame$Saccade1X[i] <=  1.48) &&
               (tempDataFrame$Saccade1Y[i] >= 0.75  && tempDataFrame$Saccade1Y[i] <=  0.95)  &&
               (tempDataFrame$Saccade1Z[i] < -6.9)) {
              tempDataFrame$QuestionId[i] <- 3
              q3XSaccads <- append(q3XSaccads, tempDataFrame$SaccadesDiffX[i])
              indexQ3    <- append(indexQ3, i)
            }
            
            oldPageIndex <- tempDataFrame$CurrentPageNumbe[i]
          }
        }
      }
      
      assign(dataframe_name, tempDataFrame)
    }
  }
  
  
  #   12.5 Save data frames as backup
  path <- file.path(condition,  "RResults", "/")
  
  pathCSV <- file.path(path,  "PagesTIMESUMsStage1_DataFrame.csv", "")
  pathTXT <- file.path(path,  "PagesTIMESUMsStage1_DataFrame.txt", "")
  write.csv2(pagesTIMESUMsStage1, pathCSV, row.names = FALSE)
  write.table(pagesTIMESUMsStage1, pathTXT, sep=" # ", row.names=FALSE)
  
  pathCSV <- file.path(path,  "PagesTIMESUMsStage2_DataFrame.csv", "")
  pathTXT <- file.path(path,  "PagesTIMESUMsStage2_DataFrame.txt", "")
  write.csv2(pagesTIMESUMsStage2, pathCSV, row.names = FALSE)
  write.table(pagesTIMESUMsStage2, pathTXT, sep=" # ", row.names=FALSE)
  
  
  for (cid in condition_list){ 
    for(stage in c("Stage0","Stage1", "Stage2")){
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      file_name_csv <- paste(dataframe_name, ".csv", sep="")
      file_name_txt <- paste(dataframe_name, ".txt", sep="")
      pathCSV <- file.path(path,  file_name_csv, "")
      pathTXT <- file.path(path,  file_name_txt, "")
      
      tempDataFrame <- get(dataframe_name)
      
      write.csv2(tempDataFrame, pathCSV, row.names = FALSE)
      write.table(tempDataFrame, pathTXT, sep=" # ", row.names=FALSE)
    }
  }
  
  
  
  #--------------------------------------------------
  # 13. Data-Fusion for all Probands in one condition
  for(stage in c("Stage0","Stage1", "Stage2")){
    all_participent_dataframe <- NULL
    for (cid in condition_list){ 
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      
      if (is.null(all_participent_dataframe)){
        all_participent_dataframe <- get(dataframe_name)
      } else {
        all_participent_dataframe <- rbind(all_participent_dataframe, get(dataframe_name))
      }
    }
    
    path <- file.path(condition,  "RResults", "/")
    file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
    pathCSV <- file.path(path, file_name, "")
    write.csv2(all_participent_dataframe, pathCSV, row.names = FALSE)
    file_name <- paste("All_Participents_", stage, "_DataFrame.txt", sep="")
    pathTXT <- file.path(path, file_name, "")
    write.table(all_participent_dataframe, pathTXT, sep=" # ", row.names=FALSE)
  }
  
  # 12.1 cleanup global environment
  id <- NULL
  condition <- NULL
  
  rm(all_pariticipent_dataframe)
  rm(all_participent_dataframe)
  
  rm(transformedBitalinoEDASiemensDataFrameStage0)
  rm(transformedBitalinoEDASiemensDataFrameStage1)
  rm(transformedBitalinoEDASiemensDataFrameStage2)
  
  rm(transformedBitalinoECGHRVDataFrameStage0)
  rm(transformedBitalinoECGHRVDataFrameStage1)
  rm(transformedBitalinoECGHRVDataFrameStage2)
  
  rm(centerOfViewInformationDataFrameStage1)
  rm(centerOfViewInformationDataFrameStage1Temp)
  rm(centerOfViewInformationStage1)
  
  rm(pagesTIMESUMsStage1)
  rm(pagesTIMESUMsStage2)
  rm(pagesTIMESUMsStage2Temp)
  rm(pagesTIMESUMs)
  rm(tempDataFrame)
  
  for (cid in condition_list){ 
    for(stage in c("Stage0","Stage1", "Stage2")){
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      remove(list=c(dataframe_name))
    }
  }
  
  rm(c_index)
  rm(lastRow)
  rm(file_name_csv)
  rm(dataframe_name)
  rm(file_name_txt)
  rm(file_name)
  rm(pagesTIMESUMs_name)
  rm(stage)
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
}
# -------------------------------------- End of one Condition fusion -----------------------------------


# 13.2 create global time rsi result of stage 1 and stage 2 for a better evaluation value

pagesTIMESUMsStage1CA <- read.csv2(file = './Condition A/RResults/PagesTIMESUMsStage1_DataFrame.csv')
pagesTIMESUMsStage1CB <- read.csv2(file = './Condition B/RResults/PagesTIMESUMsStage1_DataFrame.csv')
pagesTIMESUMsStage1CC <- read.csv2(file = './Condition C/RResults/PagesTIMESUMsStage1_DataFrame.csv')
allpagesTIMESUMsStage1 <- pagesTIMESUMsStage1CA
allpagesTIMESUMsStage1 <- rbind(allpagesTIMESUMsStage1, pagesTIMESUMsStage1CB)
allpagesTIMESUMsStage1 <- rbind(allpagesTIMESUMsStage1, pagesTIMESUMsStage1CC)

# stage 1 calc
medianTimeRsi <- 0
allpagesTIMESUMsStage1$GlobalTIMERSICalc <- 0
allpagesTIMESUMsStage1$GlobalMEDIANForTRSI <- 0
allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc <- 0
medianTimeRsi <- median(allpagesTIMESUMsStage1$TIME_SUM)
for(i in 1:nrow(allpagesTIMESUMsStage1)){
  allpagesTIMESUMsStage1$GlobalTIMERSICalc[i] <- medianTimeRsi / allpagesTIMESUMsStage1$TIME_SUM[i]
}
allpagesTIMESUMsStage1$GlobalMEDIANForTRSI <- medianTimeRsi
for(i in 1:nrow(allpagesTIMESUMsStage1)){
  if(allpagesTIMESUMsStage1$GlobalTIMERSICalc[i] < 1){
    allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i] <- 0   
  }
  if(allpagesTIMESUMsStage1$GlobalTIMERSICalc[i] >= 1){
    allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i] <- 1
  }
  if(allpagesTIMESUMsStage1$GlobalTIMERSICalc[i] >= 2){
    allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i] <- 2
  }
}

pagesTIMESUMsStage1CA$GlobalTIMERSICalc <- 0
pagesTIMESUMsStage1CA$GlobalMEDIANForTRSI <- 0
pagesTIMESUMsStage1CA$EvaluatedGlobalTIMERSICalc <- 0

pagesTIMESUMsStage1CB$GlobalTIMERSICalc <- 0
pagesTIMESUMsStage1CB$GlobalMEDIANForTRSI <- 0
pagesTIMESUMsStage1CB$EvaluatedGlobalTIMERSICalc <- 0

pagesTIMESUMsStage1CC$GlobalTIMERSICalc <- 0
pagesTIMESUMsStage1CC$GlobalMEDIANForTRSI <- 0
pagesTIMESUMsStage1CC$EvaluatedGlobalTIMERSICalc <- 0

countCa <- nrow(pagesTIMESUMsStage1CA)
countCB <- nrow(pagesTIMESUMsStage1CA) + nrow(pagesTIMESUMsStage1CB)
countCC <- nrow(allpagesTIMESUMsStage1)
  
for(i in 1:nrow(allpagesTIMESUMsStage1)){
  if(i <= countCa){
    pagesTIMESUMsStage1CA$GlobalTIMERSICalc[i] <- allpagesTIMESUMsStage1$GlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CA$EvaluatedGlobalTIMERSICalc[i] <- allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CA$GlobalMEDIANForTRSI[i] <- allpagesTIMESUMsStage1$GlobalMEDIANForTRSI[i]
  }else if(i <= countCB){
    pagesTIMESUMsStage1CB$GlobalTIMERSICalc[i - countCa] <- allpagesTIMESUMsStage1$GlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CB$EvaluatedGlobalTIMERSICalc[i - countCa] <- allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CB$GlobalMEDIANForTRSI[i - countCa] <- allpagesTIMESUMsStage1$GlobalMEDIANForTRSI[i]
  }else if(i <= countCC){
    pagesTIMESUMsStage1CC$GlobalTIMERSICalc[i - countCB] <- allpagesTIMESUMsStage1$GlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CC$EvaluatedGlobalTIMERSICalc[i - countCB] <- allpagesTIMESUMsStage1$EvaluatedGlobalTIMERSICalc[i]
    pagesTIMESUMsStage1CC$GlobalMEDIANForTRSI[i- countCB] <- allpagesTIMESUMsStage1$GlobalMEDIANForTRSI[i]
  }
}

pathCSV <- './Condition A/RResults/PagesTIMESUMsStage1_DataFrame.csv'
pathTXT <- './Condition A/RResults/PagesTIMESUMsStage1_DataFrame.txt'
write.csv2(pagesTIMESUMsStage1CA, pathCSV, row.names = FALSE)
write.table(pagesTIMESUMsStage1CA, pathTXT, sep=" # ", row.names=FALSE)

pathCSV <- './Condition B/RResults/PagesTIMESUMsStage1_DataFrame.csv'
pathTXT <- './Condition B/RResults/PagesTIMESUMsStage1_DataFrame.txt'
write.csv2(pagesTIMESUMsStage1CB, pathCSV, row.names = FALSE)
write.table(pagesTIMESUMsStage1CB, pathTXT, sep=" # ", row.names=FALSE)

pathCSV <- './Condition C/RResults/PagesTIMESUMsStage1_DataFrame.csv'
pathTXT <- './Condition C/RResults/PagesTIMESUMsStage1_DataFrame.txt'
write.csv2(pagesTIMESUMsStage1CC, pathCSV, row.names = FALSE)
write.table(pagesTIMESUMsStage1CC, pathTXT, sep=" # ", row.names=FALSE)

for (condition in c("Condition A", "Condition B", "Condition C")){
  condition_list <- NULL
  if (str_detect(condition, "Condition A")){
    condition_list <- c("id-1", "id-2", "id-3", "id-4", "id-5", "id-6", "id-7", "id-10")
  } else if (str_detect(condition, "Condition B")){
    condition_list <- c("id-13", "id-14", "id-15", "id-16", "id-17b", "id-18", "id-19", "id-20", "id-31", "id-34")
  } else if (str_detect(condition, "Condition C")){
    condition_list <- c("id-21", "id-22", "id-23", "id-24", "id-25", "id-26", "id-27", "id-28", "id-29")
  }
  
  path <- file.path(condition,  "RResults", "/")
  c_index <- 1
  tempDataFrame <- NULL
  all_participent_dataframe <- NULL
  for (cid in condition_list){ 
    for(stage in c("Stage1")){
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="") 
      file_name_csv <- paste(dataframe_name, ".csv", sep="")
      pathCSV <- file.path(path,  file_name_csv, "")
      tempDataFrame <- read.csv2(file = pathCSV)
      pagesTIMESUMs <- NULL
      if(str_detect(condition, "Condition A")){
        pagesTIMESUMs <- pagesTIMESUMsStage1CA
      }
      if(str_detect(condition, "Condition B")){
        pagesTIMESUMs <- pagesTIMESUMsStage1CB
      }
      if(str_detect(condition, "Condition C")){
        pagesTIMESUMs <- pagesTIMESUMsStage1CB
      }
      tempDataFrame$GlobalTIMERSICalc          <- pagesTIMESUMs$GlobalTIMERSICalc[c_index]
      tempDataFrame$EvaluatedGlobalTIMERSICalc <- pagesTIMESUMs$EvaluatedGlobalTIMERSICalc[c_index]
      tempDataFrame$GlobalMEDIANForTRSI        <- pagesTIMESUMs$GlobalMEDIANForTRSI[c_index]
    }
    file_name_csv <- paste(dataframe_name, ".csv", sep="")
    file_name_txt <- paste(dataframe_name, ".txt", sep="")
    pathCSV <- file.path(path,  file_name_csv, "")
    pathTXT <- file.path(path,  file_name_txt, "")
    
    write.csv2(tempDataFrame, pathCSV, row.names = FALSE)
    write.table(tempDataFrame, pathTXT, sep=" # ", row.names=FALSE)
    
    if (is.null(all_participent_dataframe)){
      all_participent_dataframe <- tempDataFrame
    } else {
      all_participent_dataframe <- rbind(all_participent_dataframe, tempDataFrame)
    }
    
    c_index <- c_index + 1
  }
  path <- file.path(condition,  "RResults", "/")
  file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path(path, file_name, "")
  write.csv2(all_participent_dataframe, pathCSV, row.names = FALSE)
  file_name <- paste("All_Participents_", stage, "_DataFrame.txt", sep="")
  pathTXT <- file.path(path, file_name, "")
  write.table(all_participent_dataframe, pathTXT, sep=" # ", row.names=FALSE)
}

# -------------------------------------- End of one extension to gobal time rsi value fusion -----------------------------------


# 14 Fuse all conditions dataframes

for(stage in c("Stage0","Stage1", "Stage2")){
  # load files
  file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition A/RResults/", file_name, "")
  conditionADataFrame <- read.csv2(file = pathCSV)
  
  file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition B/RResults/", file_name, "")
  conditionBDataFrame <- read.csv2(file = pathCSV)
  
  file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition C/RResults/", file_name, "")
  conditionCDataFrame <- read.csv2(file = pathCSV)
  
  # fuse all 3 conditions
  all_participent_dataframe <- conditionADataFrame
  all_participent_dataframe <- rbind(all_participent_dataframe, conditionBDataFrame)
  all_participent_dataframe <- rbind(all_participent_dataframe, conditionCDataFrame)
  # create a copy of fused dataframe
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
  
  # filter: remove unused/bad/const_to_zero/strings  columns
  temp_dataframe <- subset(temp_dataframe, select=-c(lex))
  if (stage != "Stage0"){
    temp_dataframe <- subset(temp_dataframe, select=-c(STARTED, LASTDATA, MAXPAGE, DegTimeThreshold, DegTimeThresholdForOnePage, DegTimeValueForOnePage)) # MAXPAGE is optional
  }
  
  # convert all columns to numeric
  numberOfColumns <- ncol(temp_dataframe)
  temp_dataframe[,1:numberOfColumns] <- lapply(temp_dataframe[,1:numberOfColumns], function (x) as.numeric(x))
  
  # save file
  file_name <- paste("All_Participents_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./", file_name, "")
  write.csv2(temp_dataframe, pathCSV, row.names = FALSE)
  
  # clean up
  rm(all_participent_dataframe)
  rm(conditionADataFrame)
  rm(conditionBDataFrame)
  rm(conditionCDataFrame)
  rm(temp_dataframe)
  rm(list)
}


# 14.1 create fuse mean data frame all participants

for (condition in c("Condition A", "Condition B", "Condition C")){
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
  path <- file.path(condition,  "RResults", "/")
  
  for(stage in c("Stage0","Stage1", "Stage2")){
    All_Participents_Mean_DataFrame <- NULL
    for (cid in condition_list){ 
      dataframe_name <- paste("participent_", cid, "_", stage, "_DataFrame", sep="")
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
      
      # filter: remove unused/bad/const_to_zero/strings  columns
      temp_dataframe <- subset(temp_dataframe, select=-c(lex))
      if (stage != "Stage0"){
        temp_dataframe <- subset(temp_dataframe, select=-c(STARTED, LASTDATA, MAXPAGE, DegTimeThreshold, DegTimeThresholdForOnePage, DegTimeValueForOnePage)) # MAXPAGE is optional
      }
      
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
    
    file_name <- paste ("All_Participents_Mean_", stage, "_DataFrame.csv", sep="")
    path_file <- file.path(path,  file_name, "")
    write.csv2(All_Participents_Mean_DataFrame, path_file, row.names = FALSE)
  }
}


# 14.2.1 create complete fuse mean data frame all participants

for(stage in c("Stage0","Stage1", "Stage2")){
  # load files
  file_name <- paste("All_Participents_Mean_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition A/RResults/", file_name, "")
  conditionAMeanDataFrame <- read.csv2(file = pathCSV)
  file_name <- paste("All_Participents_Mean_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition B/RResults/", file_name, "")
  conditionBMeanDataFrame <- read.csv2(file = pathCSV)
  file_name <- paste("All_Participents_Mean_", stage, "_DataFrame.csv", sep="")
  pathCSV <- file.path("./Condition C/RResults/", file_name, "")
  conditionCMeanDataFrame <- read.csv2(file = pathCSV)
  # fuse data frames
  all_participent_Mean_dataframes <- conditionAMeanDataFrame
  all_participent_Mean_dataframes <- rbind(all_participent_Mean_dataframes, conditionBMeanDataFrame)
  all_participent_Mean_dataframes <- rbind(all_participent_Mean_dataframes, conditionCMeanDataFrame)
  # save fused data frame
  file_name <- paste("All_Participents_Mean_", stage, "_DataFrame.csv", sep="")
  write.csv2(all_participent_Mean_dataframes, file_name, row.names = FALSE)
}

# 14.2.2 cleanup global environment

rm(all_participent_Mean_dataframes)
rm(All_Participents_Mean_DataFrame)
rm(conditionAMeanDataFrame)
rm(conditionBMeanDataFrame)
rm(conditionCMeanDataFrame)
rm(temp_dataframe)
rm(temp_df)
rm(cid)
rm(condition)
rm(condition_length)
rm(condition_list)
rm(dataframe_name)
rm(file_name)
rm(meanArray)
rm(path)
rm(path_file)
rm(numberOfColumns)
rm(pathCSV)
rm(stage)

# 14.3 create diff data frame from stage1 mean with stage0 mean data frame
# load files
file_name <- paste("All_Participents_Mean_Stage0_DataFrame.csv", sep="")
meanStage0DataFrame <- read.csv2(file = file_name)
file_name <- paste("All_Participents_Mean_Stage1_DataFrame.csv", sep="")
meanStage1DataFrame <- read.csv2(file = file_name)

cnames <- colnames(meanStage0DataFrame[,3:ncol(meanStage0DataFrame)])
allcnames <- colnames(meanStage1DataFrame)
desired_length = length(allcnames) - length(cnames)
filtertcnames <- vector(mode = "list", length = desired_length)
filtertcnames <- NULL

index <- 1
for (name in allcnames){
  if(!(name %in% cnames)){
    filtertcnames[index] <- name
    index <- index + 1
  }
}

temp_df <- meanStage1DataFrame[, cnames] - meanStage0DataFrame[, cnames]
temp_df <- cbind(meanStage1DataFrame[, filtertcnames], temp_df)

file_name <- paste("All_Participents_Mean_Diff_Of_Stages_DataFrame.csv", sep="")
write.csv2(temp_df, file_name, row.names = FALSE)

# 14.4 last cleanup stage
rm(allpagesTIMESUMsStage1)
rm(eyeTrackingSaccadesInformationStage1)
rm(meanStage0DataFrame)
rm(meanStage1DataFrame)
rm(pagesTIMESUMs)
rm(pagesTIMESUMsStage1CA)
rm(pagesTIMESUMsStage1CB)
rm(pagesTIMESUMsStage1CC)
rm(rawEyeTrackingSaccadesPositionsInformationStage1)
rm(rawFixationSaccadesPositionsDataStage1)
rm(temp_df)
rm(tempDataFrame)
rm(allcnames)
rm(c_index)
rm(countCa)
rm(countCB)
rm(countCC)
rm(file_name)
rm(file_name_csv)
rm(file_name_txt)
rm(filtertcnames)
rm(i)
rm(index)
rm(indexQ1)
rm(indexQ2)
rm(indexQ3)
rm(name)
rm(oldPageIndex)
rm(cnames)
rm(desired_length)
rm(pathTXT)
rm(medianTimeRsi)
rm(rawEyeTrackingSaccadesPositionsInformationStage1Temp)
rm(rawSaccadsPositionsInformationsAsMessage)
rm(replacedSaccade_0_Values)
rm(replacedSaccade_1_Values)
rm(q1XSaccads)
rm(q2XSaccads)
rm(q3XSaccads)
rm(rowModelName)
rm(rowModelNameIndex)
rm(saccade_0_position)
rm(saccade_1_position)
rm(splittedSaccadsPositionsInformations)


#--------------------------------------------------
# 15. Run Cluster-Algorithm en
# 15.1 Load it with python as Data Frame
# 15.2 Run and test two selected cluster algorithm en
# 15.2.1 K-Means-Clustering
# 15.2.2 Gaussian-Mixtures-Clustering

#--------------------------------------------------
# 16. Run if there are results ANOVA
# 16.1 Display the covarianz-Matrix of all features
# 16.2 Run ANOVA with covarianz results features
# 16.3 Run ANOVA with the rest features

