#Header
# position v0 (x,y,z)
# position v1 (x,y,z)
#Value
#SaccadesPositions_SaccadesTrackerReplay: "(0.124, 1.2, -7); (0.3, 1.6, -7);

# splitter is ";" and ","
 

#-------------- Stage 1
rawFixationSaccadesPositionsDataStage1 <- rawTrackingData[["Stage1"]][["CameraTracking25304"]]

rawEyeTrackingSaccadesPositionsInformationStage1 <- NULL
rawEyeTrackingSaccadesPositionsInformationStage1Temp <- NULL

for(i in 1:nrow(rawFixationSaccadesPositionsDataStage1)) {
  rowTimeValue <- rawFixationSaccadesPositionsDataStage1$time[i]
  rawSaccadsPositionsInformationsAsMessage <- rawFixationSaccadesPositionsDataStage1$SaccadesPositions_SaccadesTrackerReplay[i]

  if (!is.null(rawSaccadsPositionsInformationsAsMessage) && !is.na(rawSaccadsPositionsInformationsAsMessage) && 
      !is.nan(rawSaccadsPositionsInformationsAsMessage) && length(rawSaccadsPositionsInformationsAsMessage) && rawSaccadsPositionsInformationsAsMessage != "")
  {
    splittedSaccadsPositionsInformations <- stringr::str_split(rawSaccadsPositionsInformationsAsMessage, ";")[[1]]
    
    replacedSaccade_0_Values <- stringr::str_replace(splittedSaccadsPositionsInformations[1], "\\(", "")
    replacedSaccade_0_Values <- stringr::str_replace(replacedSaccade_0_Values, "\\)", "")
    saccade_0_position <- stringr::str_split(replacedSaccade_0_Values, ",")[[1]]
    
    replacedSaccade_1_Values <- stringr::str_replace(splittedSaccadsPositionsInformations[2], "\\(", "")
    replacedSaccade_1_Values <- stringr::str_replace(replacedSaccade_1_Values,  "\\)", "")
    saccade_1_position <- stringr::str_split(replacedSaccade_1_Values, ",")[[1]]
    
    
    if (is.null(rawEyeTrackingSaccadesPositionsInformationStage1))
    {
      p0 <- c(as.numeric(saccade_0_position[1]), as.numeric(saccade_0_position[2]), as.numeric(saccade_0_position[3]))
      if (p0[3] <= -7.0) { p0 <- (7.0/abs(p0[3])) * p0 }
      
      p1 <- c(as.numeric(saccade_1_position[1]), as.numeric(saccade_1_position[2]), as.numeric(saccade_1_position[3]))
      if (p1[3] <= -7.0) { p1 <- (7.0/abs(p1[3])) * p1 }
      
      rawEyeTrackingSaccadesPositionsInformationStage1 <- data.frame("time"       = c(as.numeric(rowTimeValue)), 
                                                                     "Saccade0X"  = c(as.numeric(p0[1])), 
                                                                     "Saccade0Y"  = c(as.numeric(p0[2])), 
                                                                     "Saccade0Z"  = c(as.numeric(p0[3])), 
                                                                     "Saccade1X"  = c(as.numeric(p1[1])), 
                                                                     "Saccade1Y"  = c(as.numeric(p1[2])), 
                                                                     "Saccade1Z"  = c(as.numeric(p1[3])),
                                                                     "QuestionId"    = c(as.numeric(0)),                   # 1 = Q-1; 2 = Q-2; 3 = Q-3 RigionOfInterest and 0 = other saccands 
                                                                     "SaccadesDiffX" = c(as.numeric(0.0)),
                                                                     "SaccadesMeanX" = c(as.numeric(0.0)),
                                                                     "SaccadesSdX"   = c(as.numeric(0.0)),
                                                                     "SaccadesMinX"  = c(as.numeric(0.0)),
                                                                     "SaccadesMaxX"  = c(as.numeric(0.0)));
    }
    else
    {
      p0 <- c(as.numeric(saccade_0_position[1]), as.numeric(saccade_0_position[2]), as.numeric(saccade_0_position[3]))
      if (p0[3] <= -7.0) { p0 <- (7.0/abs(p0[3])) * p0 }
      
      p1 <- c(as.numeric(saccade_1_position[1]), as.numeric(saccade_1_position[2]), as.numeric(saccade_1_position[3]))
      if (p1[3] <= -7.0) { p1 <- (7.0/abs(p1[3])) * p1 }
      
      row <- c(as.numeric(rowTimeValue), 
               as.numeric(p0[1]), 
               as.numeric(p0[2]), 
               as.numeric(p0[3]), 
               as.numeric(p1[1]), 
               as.numeric(p1[2]), 
               as.numeric(p1[3]), 
               as.numeric(0.0),
               as.numeric(0.0),
               as.numeric(0.0),
               as.numeric(0.0),
               as.numeric(0.0),
               as.numeric(0.0));
      
      rawEyeTrackingSaccadesPositionsInformationStage1Temp <- rawEyeTrackingSaccadesPositionsInformationStage1                   
      rawEyeTrackingSaccadesPositionsInformationStage1Temp[nrow(rawEyeTrackingSaccadesPositionsInformationStage1) + 1, ] <- row
      rawEyeTrackingSaccadesPositionsInformationStage1 <- rawEyeTrackingSaccadesPositionsInformationStage1Temp
    }
  }
}

rawEyeTrackingSaccadesPositionsInformationStage1Temp <- NULL
rowTimeValue <- NULL
rawSaccadsPositionsInformationsAsMessage <- NULL
splittedSaccadsPositionsInformations <- NULL
replacedSaccade_0_Values <- NULL
saccade_0_position <- NULL
replacedSaccade_1_Values <- NULL
saccade_1_position <- NULL
row <- NULL

