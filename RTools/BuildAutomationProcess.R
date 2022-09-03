# automation build process

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
  }
  if(grepl(id, file) && grepl("Stage-1", file) && grepl("ECG_HearRateResults", file)){
    transformedBilinoECGDataFrameStage1 <- ImporttransformedBilinoTrackingData(file)
  }
  if(grepl(id, file) && grepl("Stage-2", file) && grepl("ECG_HearRateResults", file)){
    transformedBilinoECGDataFrameStage2 <- ImporttransformedBilinoTrackingData(file)
  }
  if(grepl(id, file) && grepl("Stage-0", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage0 <- ImporttransformedBilinoTrackingData(file)
  }
  if(grepl(id, file) && grepl("Stage-1", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage1 <- ImporttransformedBilinoTrackingData(file)
  }
  if(grepl(id, file) && grepl("Stage-2", file) && grepl("EDA_EdaResults", file)){
    transformedBilinoEDADataFrameStage2 <- ImporttransformedBilinoTrackingData(file)
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
performanceMetricFrameStage1     <- downsampling(4,1)
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
#   7.1 Downsampling/Upsampling
source("DownsampleToBitalinoResults.r", echo=TRUE)
#VRQuestionnaireToolkitUncannyValleyDataStage1 <- downsampling(7,1)

# 8. RawVRQuestionnaireToolkitSSQDataFrame
source("RawVRQuestionnaireToolkitSSQDataFrame.r", echo=TRUE)
#   8.1 Downsampling to Seconds
#   8.2 Calculate Quality Values

# 9. Clear environment
source("CleanUpEnvironmentFromTemporaryUsedVariables.r", echo=TRUE)

#--------------------------------------------------
# 10. Data-Fusion of one Participant

#--------------------------------------------------
# 11. Data-Fusion for all Probands

#--------------------------------------------------
# 12. Run Cluster-Algorithmen

#--------------------------------------------------
# 13. Run if there are results ANOVA



