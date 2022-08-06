#Header
# Message  structure: STARTED; LASTDATA; LASTPAGE; MAXPAGE; FINISHED; TIME_SUM; TIME_RSI; MISSING; DEG_TIME; 
#                     DegTimeThreshold; DegTimeLowQuality; DegTimeThresholdForOnePage; DegTimeValueForOnePage; CurrentPageNumber; StandardDeviationStraightLineAnswer; AbsoluteDerivationOfResponseValue
#Value
# QualityParameters: {0}; {1}; {2}; {3}; {4}; {5}; {6}; {7}; {8}; {9}; {10}; {11}; {12}; {13}; {14}; {15};

#QualityParameters: 8/5/2022 10:37:31 PM; 8/5/2022 10:40:04 PM; 16; 18; 1; 131.1861; 0; 0; 300; 100; True; 15; 20; 17; 1; 0;

# splitter is ";" and " "

rawQualityParametersPages28114Stage1 <- rawTrackingData[["Stage1"]][["Pages28114"]]

#rawQualityParametersPages28114DataFrameStage1 <- data.frame("time" = c(as.numeric(0.0)), "STARTED" = c(0.0), "LASTDATA" = c(0.0), "LASTPAGE" = c(as.numeric(0.0)), 
#                                                            "MAXPAGE" = c(as.numeric(0.0)), "FINISHED" = c(as.numeric(0.0)), "TIME_SUM" = c(as.numeric(0.0)), 
#                                                            "TIME_RSI" = c(as.numeric(0.0)), "MISSING" = c(as.numeric(0.0)), "DEG_TIME" = c(as.numeric(0.0)),
#                                                            "DegTimeThreshold" = c(as.numeric(0.0)), "DegTimeLowQuality" = c(TRUE), "DegTimeThresholdForOnePage" = c(as.numeric(0.0)),
#                                                            "DegTimeValueForOnePage" = c(as.numeric(0.0)), "CurrentPageNumber" = c(as.numeric(0.0)),  
#                                                            "StandardDeviationStraightLineAnswer" = c(as.numeric(0.0)), "AbsoluteDerivationOfResponseValue" = c(as.numeric(0.0)));

rawQualityParametersPages28114DataFrameStage1 <- NULL

for(i in 1:nrow(rawQualityParametersPages28114Stage1)) {
  rowTimeValue <- rawQualityParametersPages28114Stage1$time[i]
  rawQualityParametersValue <- rawQualityParametersPages28114Stage1$ParametersAsMessage_PagesParameters[i]
  
  if (!is.null(rawQualityParametersValue) && !is.na(rawQualityParametersValue) && !is.nan(rawQualityParametersValue) && length(rawQualityParametersValue) && rawQualityParametersValue != "")
  {
    splittedQualityParametersValue <- stringr::str_split(rawQualityParametersValue, ";")[[1]]
    
    splittedSTARTED <- stringr::str_split(splittedQualityParametersValue[1], " ")[[1]]
    splittedSTARTED <- paste(splittedSTARTED[2], splittedSTARTED[3], splittedSTARTED[4])
    
    degTimeLowQuality <- FALSE
    if(splittedQualityParametersValue[11] == "True")
    {
      degTimeLowQuality <- TRUE
    }
    
    tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "STARTED" = c(splittedSTARTED), "LASTDATA" = c(splittedQualityParametersValue[2]), 
                                "LASTPAGE" = c(as.numeric(splittedQualityParametersValue[3])), "MAXPAGE" = c(as.numeric(splittedQualityParametersValue[4])), 
                                "FINISHED" = c(as.numeric(splittedQualityParametersValue[5])), "TIME_SUM" = c(as.numeric(splittedQualityParametersValue[6])), 
                                "TIME_RSI" = c(as.numeric(splittedQualityParametersValue[7])), "MISSING" = c(as.numeric(splittedQualityParametersValue[8])), 
                                "DEG_TIME" = c(as.numeric(splittedQualityParametersValue[9])), "DegTimeThreshold" = c(as.numeric(splittedQualityParametersValue[10])), 
                                "DegTimeLowQuality" = c(degTimeLowQuality), "DegTimeThresholdForOnePage" = c(as.numeric(splittedQualityParametersValue[12])),
                                "DegTimeValueForOnePage" = c(as.numeric(splittedQualityParametersValue[13])), "CurrentPageNumber" = c(as.numeric(splittedQualityParametersValue[14])),
                                "StandardDeviationStraightLineAnswer" = c(as.numeric(splittedQualityParametersValue[15])), "AbsoluteDerivationOfResponseValue" = c(as.numeric(splittedQualityParametersValue[16])));

    
    if (is.null(rawQualityParametersPages28114DataFrameStage1))
    {
      rawQualityParametersPages28114DataFrameStage1 <- tempDataFrame
    }
    else
    {
      rawQualityParametersPages28114DataFrameStage1 <- rbind(rawQualityParametersPages28114DataFrameStage1, tempDataFrame)
    }
  }
}
