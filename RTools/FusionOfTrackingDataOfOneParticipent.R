library(dplyr)


fuseParticipentDataFrames <- function(id, condition, stage)
{
  # print (sapply(transformedBitalinoECGDataFrameStage1, class)) # debug only
  fusedDataFrames <- NULL
  if (stage == 0){
    fusedDataFrames <- merge(transformedBitalinoECGDataFrameStage0, transformedBitalinoECGHRVDataFrameStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDADataFrameStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDASiemensDataFrameStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, bandPowerDataFrameStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, performanceMetricDataFrameStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingInformationStage0, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingSaccadesInformationStage0, by="time", all = TRUE)
  }
  if (stage == 1){
    fusedDataFrames <- merge(transformedBitalinoECGDataFrameStage1, transformedBitalinoECGHRVDataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDADataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDASiemensDataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, bandPowerDataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, performanceMetricDataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingInformationStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingSaccadesInformationStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, centerOfViewInformationDataFrameStage1, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, pagesQualityParametersStage1, by="time", all = TRUE)
  }
  if (stage == 2){
    fusedDataFrames <- merge(transformedBitalinoECGDataFrameStage2, transformedBitalinoECGHRVDataFrameStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDADataFrameStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDASiemensDataFrameStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, bandPowerDataFrameStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, performanceMetricDataFrameStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingInformationStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, eyeTrackingSaccadesInformationStage2, by="time", all = TRUE)
    fusedDataFrames <- merge(fusedDataFrames, pagesQualityParametersStage2, by="time", all = TRUE)
  }
  
  return(fusedDataFrames)
}