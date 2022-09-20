library(dplyr)


fuseParticipentDataFrames <- function(id, condition, stage)
{
  # print (sapply(transformedBitalinoECGDataFrameStage1, class)) # debug only
  # print (sapply(transformedBitalinoEDADataFrameStage1, class)) # debug only
  # print (sapply(bandPowerDataFrameStage1, class))              # debug only
  # print (sapply(performanceMetricDataFrameStage1, class))      # debug only
  # print (sapply(eyeTrackingInformationStage1, class))          # debug only
  # print (sapply(pagesQualityParametersStage1, class))          # debug only
   
  fusedDataFrames <- merge(transformedBitalinoECGDataFrameStage1, transformedBitalinoEDADataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, transformedBitalinoEDASiemensDataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, bandPowerDataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, performanceMetricDataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, eyeTrackingInformationStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, pagesQualityParametersStage1, by="time", all = TRUE)
                     
  return(fusedDataFrames)
}