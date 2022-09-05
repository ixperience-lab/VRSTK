# automation build process
library(dplyr)
library(magrittr)

# 0. Clear environment
source("CleanUpEnvironmentFromTemporaryUsedVariables.r", echo=TRUE)


# 1. ImportTrackingData
source("ImportTrackingData.r", echo=TRUE)

condition <- 'Condition B'
type_vrstk <- 'VRSTK'
type_biosppy <- 'Biosppy'
id <- 'id-13'
path <- file.path(condition,  type_vrstk, "/")

# vrstk tracking files
#rawTrackingData <- ImportTrackingData('Condition A/VRSTK/Proband-id-1-Condition-A_8-18_15-23-31.json')
vrstk_files <- list.files(path, pattern=".json", all.files=T, full.names=T)
for (file in vrstk_files) {
  if(grepl(id, file)){
    rawTrackingData <- ImportTrackingData(file)
    break
  }
}

# biosppy feature extracted file
#transformedBilinoECGDataFrameStage0 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage0 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0_id-1-Condition-A-EDA_EdaResults.txt')

#transformedBilinoECGDataFrameStage1 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage1 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-EDA_EdaResults.txt')

#transformedBilinoECGDataFrameStage2 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage2 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-EDA_EdaResults.txt')
path <- file.path(condition,  type_biosppy, "/", id, "/")
biosppy_files <- list.files(path, pattern=".txt", all.files=T, full.names=T)
for (file in biosppy_files) {
  if(grepl(id, file) && grepl("Stage-0", file) && grepl("ECG_HearRateResults", file)){
    transformedBilinoECGDataFrameStage0 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoECGDataFrameStage0) <- c('time','HeartRate','RPeaks')
    transformedBilinoECGDataFrameStage0 %<>% mutate_if(is.character, as.numeric)
  }
  if(grepl(id, file) && grepl("Stage-1", file) && grepl("ECG_HearRateResults", file)){
    transformedBilinoECGDataFrameStage1 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoECGDataFrameStage1) <- c('time','HeartRate','RPeaks')
    transformedBilinoECGDataFrameStage1 %<>% mutate_if(is.character, as.numeric)
    transformedBilinoECGDataFrameStage1$time <- as.integer(transformedBilinoECGDataFrameStage1$time)
  }
  if(grepl(id, file) && grepl("Stage-2", file) && grepl("ECG_HearRateResults", file)){
    transformedBilinoECGDataFrameStage2 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoECGDataFrameStage2) <- c('time','HeartRate','RPeaks')
    transformedBilinoECGDataFrameStage2 %<>% mutate_if(is.character, as.numeric)
  }
  if(grepl(id, file) && grepl("Stage-0", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage0 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoEDADataFrameStage0) <- c('onsets','peaks','amps')
    transformedBilinoEDADataFrameStage0 %<>% mutate_if(is.character, as.numeric)
  }
  if(grepl(id, file) && grepl("Stage-1", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage1 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoEDADataFrameStage1) <- c('onsets','peaks','amps')
    transformedBilinoEDADataFrameStage1 %<>% mutate_if(is.character, as.numeric)
  }
  if(grepl(id, file) && grepl("Stage-2", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage2 <- ImporttransformedBilinoTrackingData(file)
    colnames(transformedBilinoEDADataFrameStage2) <- c('onsets','peaks','amps')
    transformedBilinoEDADataFrameStage2 %<>% mutate_if(is.character, as.numeric)
  }
}


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
rm(participent_13_DataFrame)
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

#participent_13_DataFrame <- NULL
#participent_13_DataFrame <- fuseParticipentDataFrames(id, condition, 1)
#participent_13_DataFrame <- unique(participent_13_DataFrame)
#participent_13_Log <- duplicated(participent_13_DataFrame)
#print(participent_13_Log)
#participent_13_DataFrame <- participent_13_DataFrame %>% distinct(time, .keep_all = TRUE)


#--------------------------------------------------
# 11. Data-Fusion for all Probands
# all_pariticipent_dataframe <- NULL

# some misc cleanup
rm(nRows)
rm(countRows)
rm(participent_13_DataFrame_temp)
rm(participent_13_Log)
rm(participent_Log)

# Source: https://stackoverflow.com/questions/9368900/how-to-check-if-object-variable-is-defined-in-r
existsEnvVariable <-function(name) {
  return(1==length(ls(pattern=paste("^", name, "$", sep=""), env=globalenv())))
}


if (existsEnvVariable("all_pariticipent_dataframe"))
{
  all_pariticipent_dataframe <- rbind(all_pariticipent_dataframe, get(participent_variable_name))
} else
{
  all_pariticipent_dataframe <- get(participent_variable_name)
}

# cleanup globalenv
rm(participent_variable_name)
rm(bandPowerDataFrameStage0)
rm(bandPowerDataFrameStage1)
rm(bandPowerDataFrameStage2)

rm(eyeTrackingInformationStage0)
rm(eyeTrackingInformationStage1)
rm(eyeTrackingInformationStage2)

rm(pagesQualityParametersStage1)
rm(pagesQualityParametersStage2)

rm(performanceMetricDataFrameStage0)
rm(performanceMetricDataFrameStage1)
rm(performanceMetricDataFrameStage2)

rm(rawTrackingData)

rm(rawVRQuestionnaireToolkitSSQDataFrameStage2)
rm(rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1)
rm(tempDataFrame)

rm(transformedBilinoECGDataFrameStage0)
rm(transformedBilinoECGDataFrameStage1)
rm(transformedBilinoECGDataFrameStage2)

rm(transformedBilinoEDADataFrameStage0)
rm(transformedBilinoEDADataFrameStage1)
rm(transformedBilinoEDADataFrameStage2)

id <- NULL
condition <- NULL

# rm(all_pariticipent_dataframe)


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



