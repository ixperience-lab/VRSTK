# Header:
#   time
#   CurrentActivatedModelName_ActiveModelsReplay
# Value:
#   "time": 0.09896079, 
#   "CurrentActivatedModelName_ActiveModelsReplay": "Katie"

# Models will be saved with there index numbers as followed:
# Index_numbers:  Freddy = 0; link = 1; Gettie = 2; Pose_Zombiegirl = 3; minitrileglv1galaxy = 4; Pose_remy = 5; Lloid = 6; MedicBot = 7; Duchess = 8; ACPC_Alice = 9; Katie = 10; 
#                 Turret = 11; Eyebot = 12; ACPC_Ribbot = 13; Pose_MixamoGirl = 14; Atlas = 15

#-------------- Stage 1
centerOfViewInformationStage1 <- rawTrackingData[["Stage1"]][["CenterOfView25034"]]

centerOfViewInformationDataFrameStage1 <- NULL
centerOfViewInformationDataFrameStage1Temp <- NULL

for(i in 1:nrow(centerOfViewInformationStage1)) {
  rowTimeValue <- centerOfViewInformationStage1$time[i]
  rowModelName <- centerOfViewInformationStage1$CurrentActivatedModelName_ActiveModelsReplay[i]
  
  rowModelNameIndex <- 0
  if (grepl("link", rowModelName)){
    rowModelNameIndex <- 1 
  }
  if (grepl("Gettie", rowModelName)){
    rowModelNameIndex <- 2 
  }
  if (grepl("Pose_Zombiegirl", rowModelName)){
    rowModelNameIndex <- 3 
  }
  if (grepl("minitrileglv1galaxy", rowModelName)){
    rowModelNameIndex <- 4 
  }
  if (grepl("Pose_remy", rowModelName)){
    rowModelNameIndex <- 5 
  }
  if (grepl("Lloid", rowModelName)){
    rowModelNameIndex <- 6 
  }
  if (grepl("MedicBot", rowModelName)){
    rowModelNameIndex <- 7 
  }
  if (grepl("Duchess", rowModelName)){
    rowModelNameIndex <- 8
  }
  if (grepl("ACPC_Alice", rowModelName)){
    rowModelNameIndex <- 9
  }
  if (grepl("Katie", rowModelName)){
    rowModelNameIndex <- 10
  }
  if (grepl("Turret", rowModelName)){
    rowModelNameIndex <- 11
  }
  if (grepl("Eyebot", rowModelName)){
    rowModelNameIndex <- 12
  }
  if (grepl("ACPC_Ribbot", rowModelName)){
    rowModelNameIndex <- 13
  }
  if (grepl("Pose_MixamoGirl", rowModelName)){
    rowModelNameIndex <- 14
  }
  if (grepl("Atlas", rowModelName)){
    rowModelNameIndex <- 15
  }
  
  if (!is.null(rowModelName) && !is.na(rowModelName) && !is.nan(rowModelName) && length(rowModelName) && rowModelName != "")
  {
    if (is.null(centerOfViewInformationDataFrameStage1))
    {
      centerOfViewInformationDataFrameStage1 <- data.frame("time" = c(as.numeric(rowTimeValue)), 
                                                           "ActivatedModelIndex" = c(as.numeric(rowModelNameIndex)));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(rowModelNameIndex));
      
      centerOfViewInformationDataFrameStage1Temp <- centerOfViewInformationDataFrameStage1                   
      centerOfViewInformationDataFrameStage1Temp[nrow(centerOfViewInformationDataFrameStage1) + 1, ] <- row
      centerOfViewInformationDataFrameStage1 <- centerOfViewInformationDataFrameStage1Temp
    }
  }
}

centerOfViewInformationDataFrameStage1$time <- as.integer(centerOfViewInformationDataFrameStage1$time)

centerOfViewInformationDataFrameStage1Temp <- NULL
rowTimeValue <- NULL
rowModelName <- NULL