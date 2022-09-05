#Header
# Message  structure: STARTED; LASTDATA; LASTPAGE; MAXPAGE; FINISHED; TIME_SUM; TIME_RSI; MISSING; DEG_TIME; 
#                     DegTimeThreshold; DegTimeLowQuality; DegTimeThresholdForOnePage; DegTimeValueForOnePage; CurrentPageNumber; StandardDeviationStraightLineAnswer; AbsoluteDerivationOfResponseValue
#Value
# QualityParameters: {0}; {1}; {2}; {3}; {4}; {5}; {6}; {7}; {8}; {9}; {10}; {11}; {12}; {13}; {14}; {15};

#QualityParameters: 8/5/2022 10:37:31 PM; 8/5/2022 10:40:04 PM; 16; 18; 1; 131.1861; 0; 0; 300; 100; True; 15; 20; 17; 1; 0;

# splitter is ";" and " "

rawQualityParametersPages28114Stage1 <- rawTrackingData[["Stage1"]][["Pages28114"]]

rawQualityParametersPages28114DataFrameStage1 <- NULL
rawQualityParametersPages28114DataFrameStage1Temp <- NULL

for(i in 1:nrow(rawQualityParametersPages28114Stage1)) 
{
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
    
    if (is.null(rawQualityParametersPages28114DataFrameStage1))
    {
      tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "STARTED" = c(splittedSTARTED), "LASTDATA" = c(splittedQualityParametersValue[2]), 
                                  "LASTPAGE" = c(as.numeric(splittedQualityParametersValue[3])), "MAXPAGE" = c(as.numeric(splittedQualityParametersValue[4])), 
                                  "FINISHED" = c(as.numeric(splittedQualityParametersValue[5])), "TIME_SUM" = c(as.numeric(splittedQualityParametersValue[6])), 
                                  "TIME_RSI" = c(as.numeric(splittedQualityParametersValue[7])), "MISSING" = c(as.numeric(splittedQualityParametersValue[8])), 
                                  "DEG_TIME" = c(as.numeric(splittedQualityParametersValue[9])), "DegTimeThreshold" = c(as.numeric(splittedQualityParametersValue[10])), 
                                  "DegTimeLowQuality" = c(degTimeLowQuality), "DegTimeThresholdForOnePage" = c(as.numeric(splittedQualityParametersValue[12])),
                                  "DegTimeValueForOnePage" = c(as.numeric(splittedQualityParametersValue[13])), "CurrentPageNumber" = c(as.numeric(splittedQualityParametersValue[14])),
                                  "StandardDeviationStraightLineAnswer" = c(as.numeric(splittedQualityParametersValue[15])), "AbsoluteDerivationOfResponseValue" = c(as.numeric(splittedQualityParametersValue[16])));
      
      rawQualityParametersPages28114DataFrameStage1 <- tempDataFrame
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), splittedSTARTED, splittedQualityParametersValue[2], 
       as.numeric(splittedQualityParametersValue[3]), as.numeric(splittedQualityParametersValue[4]), 
       as.numeric(splittedQualityParametersValue[5]), as.numeric(splittedQualityParametersValue[6]), 
       as.numeric(splittedQualityParametersValue[7]), as.numeric(splittedQualityParametersValue[8]), 
       as.numeric(splittedQualityParametersValue[9]), as.numeric(splittedQualityParametersValue[10]), 
       degTimeLowQuality, as.numeric(splittedQualityParametersValue[12]),
       as.numeric(splittedQualityParametersValue[13]), as.numeric(splittedQualityParametersValue[14]),
       as.numeric(splittedQualityParametersValue[15]), as.numeric(splittedQualityParametersValue[16]));
      
      rawQualityParametersPages28114DataFrameStage1Temp <- rawQualityParametersPages28114DataFrameStage1                   
      rawQualityParametersPages28114DataFrameStage1Temp[nrow(rawQualityParametersPages28114DataFrameStage1) + 1, ] <- row
      rawQualityParametersPages28114DataFrameStage1 <- rawQualityParametersPages28114DataFrameStage1Temp
      
    }
  }
}

rawQualityParametersPages28114DataFrameStage1$time <- as.integer(rawQualityParametersPages28114DataFrameStage1$time)
rawQualityParametersPages28114DataFrameStage1$LASTPAGE <- as.numeric(rawQualityParametersPages28114DataFrameStage1$LASTPAGE)
rawQualityParametersPages28114DataFrameStage1$MAXPAGE <- as.numeric(rawQualityParametersPages28114DataFrameStage1$MAXPAGE)
rawQualityParametersPages28114DataFrameStage1$FINISHED <- as.numeric(rawQualityParametersPages28114DataFrameStage1$FINISHED)
rawQualityParametersPages28114DataFrameStage1$TIME_SUM <- as.numeric(rawQualityParametersPages28114DataFrameStage1$TIME_SUM)
rawQualityParametersPages28114DataFrameStage1$TIME_RSI <- as.numeric(rawQualityParametersPages28114DataFrameStage1$TIME_RSI)
rawQualityParametersPages28114DataFrameStage1$MISSING <- as.numeric(rawQualityParametersPages28114DataFrameStage1$MISSING)
rawQualityParametersPages28114DataFrameStage1$DEG_TIME <- as.numeric(rawQualityParametersPages28114DataFrameStage1$DEG_TIME)
rawQualityParametersPages28114DataFrameStage1$DegTimeThreshold <- as.numeric(rawQualityParametersPages28114DataFrameStage1$DegTimeThreshold)
rawQualityParametersPages28114DataFrameStage1$DegTimeThresholdForOnePage <- as.numeric(rawQualityParametersPages28114DataFrameStage1$DegTimeThresholdForOnePage)
rawQualityParametersPages28114DataFrameStage1$DegTimeValueForOnePage <- as.numeric(rawQualityParametersPages28114DataFrameStage1$DegTimeValueForOnePage)
rawQualityParametersPages28114DataFrameStage1$CurrentPageNumber <- as.numeric(rawQualityParametersPages28114DataFrameStage1$CurrentPageNumber)
rawQualityParametersPages28114DataFrameStage1$StandardDeviationStraightLineAnswer <- as.numeric(rawQualityParametersPages28114DataFrameStage1$StandardDeviationStraightLineAnswer)
rawQualityParametersPages28114DataFrameStage1$AbsoluteDerivationOfResponseValue <- as.numeric(rawQualityParametersPages28114DataFrameStage1$AbsoluteDerivationOfResponseValue)

#print (sapply(rawQualityParametersPages28114DataFrameStage1, class))


rawQualityParametersPages28114Stage2 <- rawTrackingData[["Stage2"]][["Pages28800"]]

rawQualityParametersPages28114DataFrameStage2 <- NULL
rawQualityParametersPages28114DataFrameStage2Temp <- NULL

for(i in 1:nrow(rawQualityParametersPages28114Stage2)) 
{
  rowTimeValue <- rawQualityParametersPages28114Stage2$time[i]
  rawQualityParametersValue <- rawQualityParametersPages28114Stage2$ParametersAsMessage_PagesParameters[i]
  
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
    
    if (is.null(rawQualityParametersPages28114DataFrameStage2))
    {
      tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "STARTED" = c(splittedSTARTED), "LASTDATA" = c(splittedQualityParametersValue[2]), 
                                  "LASTPAGE" = c(as.numeric(splittedQualityParametersValue[3])), "MAXPAGE" = c(as.numeric(splittedQualityParametersValue[4])), 
                                  "FINISHED" = c(as.numeric(splittedQualityParametersValue[5])), "TIME_SUM" = c(as.numeric(splittedQualityParametersValue[6])), 
                                  "TIME_RSI" = c(as.numeric(splittedQualityParametersValue[7])), "MISSING" = c(as.numeric(splittedQualityParametersValue[8])), 
                                  "DEG_TIME" = c(as.numeric(splittedQualityParametersValue[9])), "DegTimeThreshold" = c(as.numeric(splittedQualityParametersValue[10])), 
                                  "DegTimeLowQuality" = c(degTimeLowQuality), "DegTimeThresholdForOnePage" = c(as.numeric(splittedQualityParametersValue[12])),
                                  "DegTimeValueForOnePage" = c(as.numeric(splittedQualityParametersValue[13])), "CurrentPageNumber" = c(as.numeric(splittedQualityParametersValue[14])),
                                  "StandardDeviationStraightLineAnswer" = c(as.numeric(splittedQualityParametersValue[15])), "AbsoluteDerivationOfResponseValue" = c(as.numeric(splittedQualityParametersValue[16])));
      
      rawQualityParametersPages28114DataFrameStage2 <- tempDataFrame
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), splittedSTARTED, splittedQualityParametersValue[2], 
               as.numeric(splittedQualityParametersValue[3]), as.numeric(splittedQualityParametersValue[4]), 
               as.numeric(splittedQualityParametersValue[5]), as.numeric(splittedQualityParametersValue[6]), 
               as.numeric(splittedQualityParametersValue[7]), as.numeric(splittedQualityParametersValue[8]), 
               as.numeric(splittedQualityParametersValue[9]), as.numeric(splittedQualityParametersValue[10]), 
               degTimeLowQuality, as.numeric(splittedQualityParametersValue[12]),
               as.numeric(splittedQualityParametersValue[13]), as.numeric(splittedQualityParametersValue[14]),
               as.numeric(splittedQualityParametersValue[15]), as.numeric(splittedQualityParametersValue[16]));
      
      rawQualityParametersPages28114DataFrameStage2Temp <- rawQualityParametersPages28114DataFrameStage2                   
      rawQualityParametersPages28114DataFrameStage2Temp[nrow(rawQualityParametersPages28114DataFrameStage2) + 1, ] <- row
      rawQualityParametersPages28114DataFrameStage2 <- rawQualityParametersPages28114DataFrameStage2Temp
      

    }
  }
}
