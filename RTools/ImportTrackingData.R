library(psych)
library(stringi)
library(stringr)
library(jsonlite)
library(glue)
library(magrittr)
library(mnormt)

ImportTrackingData <- function(filePath)
{
  require(jsonlite);
  
  rawTrackingData <- fromJSON(filePath);
  
  return(rawTrackingData);
}

#rawTrackingData <- ImportTrackingData('Condition A/VRSTK/Proband-id-1-Condition-A_8-18_15-23-31.json')

ImporttransformedBilinoTrackingData <- function(filePath)
{
  # header_hr "Heart_Rate ; ts_Haert_Rate ; R_Peaks"
  # header_eda "onsets ; peaks ; amps"
  transformedBilinoDataFrame <- read.table(filePath, sep =";", header = TRUE, dec =".")
  
  return(transformedBilinoDataFrame);
}


#transformedBilinoECGDataFrameStage0 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage0 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-0_id-1-Condition-A-EDA_EdaResults.txt')

#transformedBilinoECGDataFrameStage1 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage1 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-1-id-1-Condition-A-EDA_EdaResults.txt')

#transformedBilinoECGDataFrameStage2 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-ECG_HearRateResults.txt')
#transformedBilinoEDADataFrameStage2 <- ImporttransformedBilinoTrackingData('Condition A/Biosppy/id-1/Bitalinoi-Proband-Stage-2-id-1-Condition-A-EDA_EdaResults.txt')



