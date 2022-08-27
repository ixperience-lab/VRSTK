# Header Left Eye:
#   timestamp, _sRanipalFrameSequence ; _leftEyeDataValidataBitMask ; _leftEyeOpenness ; _leftPupilDiameter ; 
#   _leftPupilPositionInSensorArea ; _leftGazeOrigin_mm ; _leftGazeDirectionNormalized ; 
#   _leftGazeDirectionNormalizedTranslatedToWorldSpace 
# Value:
#   EyeTrackingLeftEyeInformationsAsMessage : 3107056;9028;31;1;3.047989;(0.5, 0.6);(32.7, 5.3, -40.8);(0.0, -0.2, 1.0);(0.0, -0.1, -1.0);

# Header Right Eye:
#   timestamp ; _sRanipalFrameSequence ; _rightEyeDataValidataBitMask ; _rightEyeOpenness ; _rightPupilDiameter ; 
#   _rightPupilPositionInSensorArea ; _rightGazeOrigin_mm ; _rightGazeDirectionNormalized ; 
#   _rightGazeDirectionNormalizedTranslatedToWorldSpace 
# Value:
#   EyeTrackingRightEyeInformationsAsMessage : 3107056;9028;31;1;3.672501;(0.5, 0.5);(-29.4, 5.9, -38.9);(0.0, -0.2, 1.0);(0.0, -0.1, -1.0);

# Header Fixations Informations:
#   _totalFixationCounter ; _fixationCounter ; _totalFixationDuration ; _fixationDuration
# Value:
#   FixationsInformationAsMessage : 1227;27;28.70996;0.6499023;

# Header Saccads Informations:
#   _saccadeVelocityThreshold ; _measuredVisualAngle ; _measuredVelocity ; _saccadeCounter 
# Value:
#   SaccadsInformationsAsMessage : 70;0.5417637;21.75553;138;

# Splitter: ";"

#-------------- Stage 0
rawEyeTrackingInformationStage0 <- rawTrackingData[["Stage0"]][["CameraTracking25332"]]

rawEyeTrackingInformationDataFrameStage0 <- NULL
rawEyeTrackingInformationDataFrameStage0Temp <- NULL

for(i in 1:nrow(rawEyeTrackingInformationStage0)) {
  rowTimeValue <- rawEyeTrackingInformationStage0$time[i]
  rawLeftEyeInformationsAsMessage <- rawEyeTrackingInformationStage0$EyeTrackingLeftEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawRightEyeInformationsAsMessage <- rawEyeTrackingInformationStage0$EyeTrackingRightEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawFixationsInformationAsMessage <- rawEyeTrackingInformationStage0$FixationsInformationAsMessage_EyeTrackingCalibrationSaccades[i]
  rawSaccadsInformationsAsMessage <- rawEyeTrackingInformationStage0$SaccadsInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  
  
  if (!is.null(rawLeftEyeInformationsAsMessage) && !is.na(rawLeftEyeInformationsAsMessage) && !is.nan(rawLeftEyeInformationsAsMessage) && length(rawLeftEyeInformationsAsMessage) && rawLeftEyeInformationsAsMessage != "")
  {
    splittedLeftEyeInformationsValue <- stringr::str_split(rawLeftEyeInformationsAsMessage, ";")[[1]]
    splittedRightEyeInformationsValue <- stringr::str_split(rawRightEyeInformationsAsMessage, ";")[[1]]
    splittedFixationsInformationValue <- stringr::str_split(rawFixationsInformationAsMessage, ";")[[1]]
    splittedSaccadsInformationsValue <- stringr::str_split(rawSaccadsInformationsAsMessage, ";")[[1]]
    
    if (is.null(rawEyeTrackingInformationDataFrameStage0))
    {
      rawEyeTrackingInformationDataFrameStage0 <- data.frame("time" = c(as.numeric(rowTimeValue)), 
                                                             "LeftEyeOpenness" = c(as.numeric(splittedLeftEyeInformationsValue[3])), 
                                                             "LeftPupilDiameter" = c(as.numeric(splittedLeftEyeInformationsValue[4])), 
                                                             "RightEyeOpenness" = c(as.numeric(splittedRightEyeInformationsValue[3])), 
                                                             "RightPupilDiameter" = c(as.numeric(splittedRightEyeInformationsValue[4])), 
                                                             "TotalFixationCounter" = c(as.numeric(splittedFixationsInformationValue[1])), 
                                                             "FixationCounter" = c(as.numeric(splittedFixationsInformationValue[2])), 
                                                             "TotalFixationDuration" = c(as.numeric(splittedFixationsInformationValue[3])),
                                                             "FixationDuration" = c(as.numeric(splittedFixationsInformationValue[4])),
                                                             "MeasuredVelocity" = c(as.numeric(splittedSaccadsInformationsValue[3])),
                                                             "SaccadeCounter" = c(as.numeric(splittedSaccadsInformationsValue[4])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), 
               as.numeric(splittedLeftEyeInformationsValue[3]), 
               as.numeric(splittedLeftEyeInformationsValue[4]), 
               as.numeric(splittedRightEyeInformationsValue[3]), 
               as.numeric(splittedRightEyeInformationsValue[4]), 
               as.numeric(splittedFixationsInformationValue[1]), 
               as.numeric(splittedFixationsInformationValue[2]), 
               as.numeric(splittedFixationsInformationValue[3]),
               as.numeric(splittedFixationsInformationValue[4]),
               as.numeric(splittedSaccadsInformationsValue[3]),
               as.numeric(splittedSaccadsInformationsValue[4]));
      
      rawEyeTrackingInformationDataFrameStage0Temp <- rawEyeTrackingInformationDataFrameStage0                   
      rawEyeTrackingInformationDataFrameStage0Temp[nrow(rawEyeTrackingInformationDataFrameStage0) + 1, ] <- row
      rawEyeTrackingInformationDataFrameStage0 <- rawEyeTrackingInformationDataFrameStage0Temp
    }
  }
}

rawEyeTrackingInformationDataFrameStage0Temp <- NULL
rowTimeValue <- NULL
rawLeftEyeInformationsAsMessage <- NULL
rawRightEyeInformationsAsMessage <- NULL
rawFixationsInformationAsMessage <- NULL
rawSaccadsInformationsAsMessage <- NULL
splittedLeftEyeInformationsValue <- NULL
splittedRightEyeInformationsValue <- NULL
splittedFixationsInformationValue <- NULL
splittedSaccadsInformationsValue <-  NULL


#-------------- Stage 1
rawEyeTrackingInformationStage1 <- rawTrackingData[["Stage1"]][["CameraTracking25332"]]

rawEyeTrackingInformationDataFrameStage1 <- NULL
rawEyeTrackingInformationDataFrameStage1Temp <- NULL

for(i in 1:nrow(rawEyeTrackingInformationStage1)) {
  rowTimeValue <- rawEyeTrackingInformationStage1$time[i]
  rawLeftEyeInformationsAsMessage <- rawEyeTrackingInformationStage1$EyeTrackingLeftEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawRightEyeInformationsAsMessage <- rawEyeTrackingInformationStage1$EyeTrackingRightEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawFixationsInformationAsMessage <- rawEyeTrackingInformationStage1$FixationsInformationAsMessage_EyeTrackingCalibrationSaccades[i]
  rawSaccadsInformationsAsMessage <- rawEyeTrackingInformationStage1$SaccadsInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  
  
  if (!is.null(rawLeftEyeInformationsAsMessage) && !is.na(rawLeftEyeInformationsAsMessage) && !is.nan(rawLeftEyeInformationsAsMessage) && length(rawLeftEyeInformationsAsMessage) && rawLeftEyeInformationsAsMessage != "")
  {
    splittedLeftEyeInformationsValue <- stringr::str_split(rawLeftEyeInformationsAsMessage, ";")[[1]]
    splittedRightEyeInformationsValue <- stringr::str_split(rawRightEyeInformationsAsMessage, ";")[[1]]
    splittedFixationsInformationValue <- stringr::str_split(rawFixationsInformationAsMessage, ";")[[1]]
    splittedSaccadsInformationsValue <- stringr::str_split(rawSaccadsInformationsAsMessage, ";")[[1]]
    
    if (is.null(rawEyeTrackingInformationDataFrameStage1))
    {
      rawEyeTrackingInformationDataFrameStage1 <- data.frame("time" = c(as.numeric(rowTimeValue)), 
                                                             "LeftEyeOpenness" = c(as.numeric(splittedLeftEyeInformationsValue[3])), 
                                                             "LeftPupilDiameter" = c(as.numeric(splittedLeftEyeInformationsValue[4])), 
                                                             "RightEyeOpenness" = c(as.numeric(splittedRightEyeInformationsValue[3])), 
                                                             "RightPupilDiameter" = c(as.numeric(splittedRightEyeInformationsValue[4])), 
                                                             "TotalFixationCounter" = c(as.numeric(splittedFixationsInformationValue[1])), 
                                                             "FixationCounter" = c(as.numeric(splittedFixationsInformationValue[2])), 
                                                             "TotalFixationDuration" = c(as.numeric(splittedFixationsInformationValue[3])),
                                                             "FixationDuration" = c(as.numeric(splittedFixationsInformationValue[4])),
                                                             "MeasuredVelocity" = c(as.numeric(splittedSaccadsInformationsValue[3])),
                                                             "SaccadeCounter" = c(as.numeric(splittedSaccadsInformationsValue[4])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), 
               as.numeric(splittedLeftEyeInformationsValue[3]), 
               as.numeric(splittedLeftEyeInformationsValue[4]), 
               as.numeric(splittedRightEyeInformationsValue[3]), 
               as.numeric(splittedRightEyeInformationsValue[4]), 
               as.numeric(splittedFixationsInformationValue[1]), 
               as.numeric(splittedFixationsInformationValue[2]), 
               as.numeric(splittedFixationsInformationValue[3]),
               as.numeric(splittedFixationsInformationValue[4]),
               as.numeric(splittedSaccadsInformationsValue[3]),
               as.numeric(splittedSaccadsInformationsValue[4]));
      
      rawEyeTrackingInformationDataFrameStage1Temp <- rawEyeTrackingInformationDataFrameStage1                   
      rawEyeTrackingInformationDataFrameStage1Temp[nrow(rawEyeTrackingInformationDataFrameStage1) + 1, ] <- row
      rawEyeTrackingInformationDataFrameStage1 <- rawEyeTrackingInformationDataFrameStage1Temp
    }
  }
}

rawEyeTrackingInformationDataFrameStage1Temp <- NULL
rowTimeValue <- NULL
rawLeftEyeInformationsAsMessage <- NULL
rawRightEyeInformationsAsMessage <- NULL
rawFixationsInformationAsMessage <- NULL
rawSaccadsInformationsAsMessage <- NULL
splittedLeftEyeInformationsValue <- NULL
splittedRightEyeInformationsValue <- NULL
splittedFixationsInformationValue <- NULL
splittedSaccadsInformationsValue <-  NULL


#-------------- Stage 2
rawEyeTrackingInformationStage2 <- rawTrackingData[["Stage2"]][["CameraTracking25332"]]

rawEyeTrackingInformationDataFrameStage2 <- NULL
rawEyeTrackingInformationDataFrameStage2Temp <- NULL

for(i in 1:nrow(rawEyeTrackingInformationStage2)) {
  rowTimeValue <- rawEyeTrackingInformationStage2$time[i]
  rawLeftEyeInformationsAsMessage <- rawEyeTrackingInformationStage2$EyeTrackingLeftEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawRightEyeInformationsAsMessage <- rawEyeTrackingInformationStage2$EyeTrackingRightEyeInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  rawFixationsInformationAsMessage <- rawEyeTrackingInformationStage2$FixationsInformationAsMessage_EyeTrackingCalibrationSaccades[i]
  rawSaccadsInformationsAsMessage <- rawEyeTrackingInformationStage2$SaccadsInformationsAsMessage_EyeTrackingCalibrationSaccades[i]
  
  
  if (!is.null(rawLeftEyeInformationsAsMessage) && !is.na(rawLeftEyeInformationsAsMessage) && !is.nan(rawLeftEyeInformationsAsMessage) && length(rawLeftEyeInformationsAsMessage) && rawLeftEyeInformationsAsMessage != "")
  {
    splittedLeftEyeInformationsValue <- stringr::str_split(rawLeftEyeInformationsAsMessage, ";")[[1]]
    splittedRightEyeInformationsValue <- stringr::str_split(rawRightEyeInformationsAsMessage, ";")[[1]]
    splittedFixationsInformationValue <- stringr::str_split(rawFixationsInformationAsMessage, ";")[[1]]
    splittedSaccadsInformationsValue <- stringr::str_split(rawSaccadsInformationsAsMessage, ";")[[1]]
    
    if (is.null(rawEyeTrackingInformationDataFrameStage2))
    {
      rawEyeTrackingInformationDataFrameStage2 <- data.frame("time" = c(as.numeric(rowTimeValue)), 
                                                             "LeftEyeOpenness" = c(as.numeric(splittedLeftEyeInformationsValue[3])), 
                                                             "LeftPupilDiameter" = c(as.numeric(splittedLeftEyeInformationsValue[4])), 
                                                             "RightEyeOpenness" = c(as.numeric(splittedRightEyeInformationsValue[3])), 
                                                             "RightPupilDiameter" = c(as.numeric(splittedRightEyeInformationsValue[4])), 
                                                             "TotalFixationCounter" = c(as.numeric(splittedFixationsInformationValue[1])), 
                                                             "FixationCounter" = c(as.numeric(splittedFixationsInformationValue[2])), 
                                                             "TotalFixationDuration" = c(as.numeric(splittedFixationsInformationValue[3])),
                                                             "FixationDuration" = c(as.numeric(splittedFixationsInformationValue[4])),
                                                             "MeasuredVelocity" = c(as.numeric(splittedSaccadsInformationsValue[3])),
                                                             "SaccadeCounter" = c(as.numeric(splittedSaccadsInformationsValue[4])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), 
               as.numeric(splittedLeftEyeInformationsValue[3]), 
               as.numeric(splittedLeftEyeInformationsValue[4]), 
               as.numeric(splittedRightEyeInformationsValue[3]), 
               as.numeric(splittedRightEyeInformationsValue[4]), 
               as.numeric(splittedFixationsInformationValue[1]), 
               as.numeric(splittedFixationsInformationValue[2]), 
               as.numeric(splittedFixationsInformationValue[3]),
               as.numeric(splittedFixationsInformationValue[4]),
               as.numeric(splittedSaccadsInformationsValue[3]),
               as.numeric(splittedSaccadsInformationsValue[4]));
      
      rawEyeTrackingInformationDataFrameStage2Temp <- rawEyeTrackingInformationDataFrameStage2                   
      rawEyeTrackingInformationDataFrameStage2Temp[nrow(rawEyeTrackingInformationDataFrameStage2) + 1, ] <- row
      rawEyeTrackingInformationDataFrameStage2 <- rawEyeTrackingInformationDataFrameStage2Temp
    }
  }
}

rawEyeTrackingInformationDataFrameStage2Temp <- NULL
rowTimeValue <- NULL
rawLeftEyeInformationsAsMessage <- NULL
rawRightEyeInformationsAsMessage <- NULL
rawFixationsInformationAsMessage <- NULL
rawSaccadsInformationsAsMessage <- NULL
splittedLeftEyeInformationsValue <- NULL
splittedRightEyeInformationsValue <- NULL
splittedFixationsInformationValue <- NULL
splittedSaccadsInformationsValue <-  NULL
