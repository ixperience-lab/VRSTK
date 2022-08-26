#Header
# Data structure: 
# MISSING;
# TIME_RSI;
# TIME_SUM;
# MISSREL; 
# DEG_TIME;
# LastQuestionnaire_TIME_SUM; 
#Value
# QualityParameters: 
# 0 
# 1.0; 
# 432.5862 
# -0.0212766  
# 0.0
# 432.5862

# splitter is ";" for TIME_RSI and MISSREL

rawVRQuestionnaireToolkitSSQDataStage2 <- rawTrackingData[["Stage2"]][["VRQuestionnaireToolkitSSQ37936"]]

rawVRQuestionnaireToolkitSSQDataFrameStage2 <- NULL
rawVRQuestionnaireToolkitSSQDataFrameStage2Temp <- NULL
row <- NULL

for(i in 1:nrow(rawVRQuestionnaireToolkitSSQDataStage2)) {
  rowTimeValue <- rawVRQuestionnaireToolkitSSQDataStage2$time[i]
  rawMissingValue <- rawVRQuestionnaireToolkitSSQDataStage2$MISSING_Message_QualityParameters[i]
  rawTIMERSIValue <- rawVRQuestionnaireToolkitSSQDataStage2$TIME_RSI_Message_QualityParameters[i]
  rawTIMESUMValue <- rawVRQuestionnaireToolkitSSQDataStage2$TIME_SUM_Message_QualityParameters[i]
  rawMISSRELValue <- rawVRQuestionnaireToolkitSSQDataStage2$MISSREL_Message_QualityParameters[i]
  rawDEGTIMEValue <- rawVRQuestionnaireToolkitSSQDataStage2$DEG_TIME_Message_QualityParameters[i]
  rawLastTIMESUM <- rawVRQuestionnaireToolkitSSQDataStage2$LastQuestionnaire_TIME_SUM_QualityParameters[i]
  
  if (!is.null(rawMissingValue) && !is.na(rawMissingValue) && !is.nan(rawMissingValue) && length(rawMissingValue) && rawMissingValue != "") 
  {
    splittedTIMERSIValue <- stringr::str_split(rawTIMERSIValue, ";")[[1]]
    splittedMISSRELValue <- stringr::str_split(rawMISSRELValue, ";")[[1]]
    
    if (is.null(rawVRQuestionnaireToolkitSSQDataFrameStage2)) 
    {
      rawVRQuestionnaireToolkitSSQDataFrameStage2 <- data.frame("time" = c(as.numeric(rowTimeValue)), "MISSING" = c(as.numeric(rawMissingValue)), 
                                                                "TIME_RSI" = c(as.numeric(splittedTIMERSIValue[1])), "TIME_SUM" = c(as.numeric(rawTIMESUMValue)), 
                                                                "MISSREL" = c(as.numeric(splittedMISSRELValue[1])), "DEG_TIME" = c(as.numeric(rawDEGTIMEValue)), 
                                                                "LastQuestionnaire_TIME_SUM" = c(as.numeric(rawLastTIMESUM)));
    } 
    else 
    { 
      row <- c(as.numeric(rowTimeValue), as.numeric(rawMissingValue), 
               as.numeric(splittedTIMERSIValue[1]), as.numeric(rawTIMESUMValue), 
               as.numeric(splittedMISSRELValue[1]), as.numeric(rawDEGTIMEValue), 
               as.numeric(rawLastTIMESUM))  
      
      rawVRQuestionnaireToolkitSSQDataFrameStage2Temp <- rawVRQuestionnaireToolkitSSQDataFrameStage2                   
      rawVRQuestionnaireToolkitSSQDataFrameStage2Temp[nrow(rawVRQuestionnaireToolkitSSQDataFrameStage2) + 1, ] <- row
      rawVRQuestionnaireToolkitSSQDataFrameStage2 <- rawVRQuestionnaireToolkitSSQDataFrameStage2Temp
    }
  }
}

rawVRQuestionnaireToolkitSSQDataFrameStage2Temp <- NULL
row <- NULL
splittedTIMERSIValue <- NULL
splittedMISSRELValue <- NULL
rowTimeValue <- NULL
rawMissingValue <- NULL
rawTIMERSIValue <- NULL
rawTIMESUMValue <- NULL
rawMISSRELValue <- NULL
rawDEGTIMEValue <- NULL
rawLastTIMESUM <- NULL
