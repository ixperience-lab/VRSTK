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

rawVRQuestionnaireToolkitUncannyValleyDataStage1 <- rawTrackingData[["Stage1"]][["VRQuestionnaireToolkitUncannyValley34206"]]

rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1 <- NULL
rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1Temp <- NULL
row <- NULL

for(i in 1:nrow(rawVRQuestionnaireToolkitUncannyValleyDataStage1)) {
  rowTimeValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$time[i]
  rawMissingValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$MISSING_Message_QualityParameters[i]
  rawTIMERSIValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$TIME_RSI_Message_QualityParameters[i]
  rawTIMESUMValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$TIME_SUM_Message_QualityParameters[i]
  rawMISSRELValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$MISSREL_Message_QualityParameters[i]
  rawDEGTIMEValue <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$DEG_TIME_Message_QualityParameters[i]
  rawLastTIMESUM <- rawVRQuestionnaireToolkitUncannyValleyDataStage1$LastQuestionnaire_TIME_SUM_QualityParameters[i]
  
  if (!is.null(rawMissingValue) && !is.na(rawMissingValue) && !is.nan(rawMissingValue) && length(rawMissingValue) && rawMissingValue != "") 
  {
    splittedTIMERSIValue <- stringr::str_split(rawTIMERSIValue, ";")[[1]]
    splittedMISSRELValue <- stringr::str_split(rawMISSRELValue, ";")[[1]]
    
    if (is.null(rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1)) 
    {
      rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1 <- data.frame("time" = c(as.numeric(rowTimeValue)), "MISSING" = c(as.numeric(rawMissingValue)), 
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
      
      rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1Temp <- rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1                   
      rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1Temp[nrow(rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1) + 1, ] <- row
      rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1 <- rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1Temp
    }
  }
}

rawVRQuestionnaireToolkitUncannyValleyDataFrameStage1Temp <- NULL
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
