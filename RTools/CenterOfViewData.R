# Header:
#   time
#   CurrentActivatedModelName_ActiveModelsReplay
# Value:
#   "time": 0.09896079, 
#   "CurrentActivatedModelName_ActiveModelsReplay": "Katie"

# Models will be saved with there index numbers as followed:
# Index_numbers: 
#  
# org_indexes_from_paper: All different characters used for evaluation. 1. Eyebot, 2. Turret, 3. JRRobo, 4. Lloyd, 5. Atlas, 6. 
#                         Ribbot, 7. Katie, 8. Alice, 9. Freddy, 10. Medic, 11. Link, 12. Dutchess, 13. Zombie, 14. MixamoGirl, 15. Remy
#
# Index_numbers:  Gettie (not include in the massure of UV) = 0; Eyebot = 1; Turret = 2; minitrileglv1galaxy(JRRobo) = 3; Lloid = 4; Atlas = 5; 
#                 ACPC_Ribbot = 6; Katie = 7; ACPC_Alice = 8; Freddy = 9; MedicBot = 10; link = 11; Duchess = 12; Pose_Zombiegirl = 13; Pose_MixamoGirl = 14; Pose_remy = 15;



#-------------- Stage 1
centerOfViewInformationStage1 <- rawTrackingData[["Stage1"]][["CenterOfView25034"]]

centerOfViewInformationDataFrameStage1 <- NULL
centerOfViewInformationDataFrameStage1Temp <- NULL

for(i in 1:nrow(centerOfViewInformationStage1)) {
  rowTimeValue <- centerOfViewInformationStage1$time[i]
  rowModelName <- centerOfViewInformationStage1$CurrentActivatedModelName_ActiveModelsReplay[i]
  
  rowModelNameIndex <- 0 # Gettie index number
  
  if (grepl("Eyebot", rowModelName)){
    rowModelNameIndex <- 1
  }
  if (grepl("Turret", rowModelName)){
    rowModelNameIndex <- 2
  }
  if (grepl("minitrileglv1galaxy", rowModelName)){
    rowModelNameIndex <- 3 
  }
  if (grepl("Lloid", rowModelName)){
    rowModelNameIndex <- 4 
  }
  if (grepl("Atlas", rowModelName)){
    rowModelNameIndex <- 5
  }
  if (grepl("ACPC_Ribbot", rowModelName)){
    rowModelNameIndex <- 6
  }
  if (grepl("Katie", rowModelName)){
    rowModelNameIndex <- 7
  }
  if (grepl("ACPC_Alice", rowModelName)){
    rowModelNameIndex <- 8
  }
  if (grepl("Freddy", rowModelName)){
    rowModelNameIndex <- 9 
  }
  if (grepl("MedicBot", rowModelName)){
    rowModelNameIndex <- 10 
  }
  if (grepl("link", rowModelName)){
    rowModelNameIndex <- 11 
  }
  if (grepl("Duchess", rowModelName)){
    rowModelNameIndex <- 12
  }
  if (grepl("Pose_Zombiegirl", rowModelName)){
    rowModelNameIndex <- 13 
  }
  if (grepl("Pose_MixamoGirl", rowModelName)){
    rowModelNameIndex <- 14
  }
  if (grepl("Pose_remy", rowModelName)){
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