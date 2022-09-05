library(dplyr)


fuseParticipentDataFrames <- function(id, condition, stage)
{
  print (sapply(transformedBilinoECGDataFrameStage1, class))
  print (sapply(bandPowerDataFrameStage1, class))
  print (sapply(performanceMetricDataFrameStage1, class))
  print (sapply(eyeTrackingInformationStage1, class))
  print (sapply(pagesQualityParametersStage1, class))
  
  #fusedDataFrames <- transformedBilinoECGDataFrameStage1 %>% 
  #                   left_join(bandPowerDataFrameStage1, by='time') %>% 
  #                   left_join(performanceMetricDataFrameStage1, by='time')  %>%
  #                   left_join(eyeTrackingInformationStage1, by='time')  %>%
  #                   left_join(pagesQualityParametersStage1, by='time')  
                     #%>% left_join(UncannyValleyQualityPrameters, by='time')
  
  #fusedDataFrames <- cbind(transformedBilinoECGDataFrameStage1, bandPowerDataFrameStage1)
  
  #fusedDataFrames <- merge(transformedBilinoECGDataFrameStage1, bandPowerDataFrameStage1, by="time")
   
  fusedDataFrames <- merge(transformedBilinoECGDataFrameStage1, bandPowerDataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, performanceMetricDataFrameStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, eyeTrackingInformationStage1, by="time", all = TRUE)
  fusedDataFrames <- merge(fusedDataFrames, pagesQualityParametersStage1, by="time", all = TRUE)
                     
  return(fusedDataFrames)
}