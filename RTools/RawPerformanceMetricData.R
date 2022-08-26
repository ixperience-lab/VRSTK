#Header
#"eng.isActive","eng",
#"exc.isActive","exc","lex",
#"str.isActive","str",
#"rel.isActive","rel",
#"int.isActive","int",
#"foc.isActive","foc"
#Value
#"met":[false,null,false,null,null,false,null,true,0.266589,false,null,true,0.098421],
#"sid":"6a68b92a-cb1f-4062-bf1f-74424fbae065",
#"time":1559903137.1741

#met data: 1659612205,5648;True;0,572177;True;0,484655;0;True;0,285521;True;0,316559;True;0,413041;True;0,483939;

# splitter is ";"

#-------------- Stage 0
rawPerformanceMetricDataFrameStage0 <- NULL
rawPerformanceMetricDataFrameStage0Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage0)) {
  rowTimeValue <- rawEmotivTrackingDataStage0$time[i]
  rawPerformanceMetricValue <- rawEmotivTrackingDataStage0$PerformanceMetricsDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rawPerformanceMetricValue) && !is.na(rawPerformanceMetricValue) && !is.nan(rawPerformanceMetricValue) && length(rawPerformanceMetricValue) && rawPerformanceMetricValue != "")
  {
    splittedPerformanceMetricValue <- stringr::str_split(rawPerformanceMetricValue, ";")[[1]]
    splittedPerformanceMetricValue <- stringr::str_replace(splittedPerformanceMetricValue, ",", ".")
    
    if (is.null(rawPerformanceMetricDataFrameStage0))
    {
      rawPerformanceMetricDataFrameStage0 <- data.frame("time" = c(as.numeric(rowTimeValue)), "eng" = c(as.numeric(splittedPerformanceMetricValue[3])), "exc" = c(as.numeric(splittedPerformanceMetricValue[5])), 
                                  "lex" = c(as.numeric(splittedPerformanceMetricValue[6])), "str" = c(as.numeric(splittedPerformanceMetricValue[8])), "rel" = c(as.numeric(splittedPerformanceMetricValue[10])), 
                                  "int" = c(as.numeric(splittedPerformanceMetricValue[12])), "foc" = c(as.numeric(splittedPerformanceMetricValue[14])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(splittedPerformanceMetricValue[3]), as.numeric(splittedPerformanceMetricValue[5]), 
               as.numeric(splittedPerformanceMetricValue[6]),  as.numeric(splittedPerformanceMetricValue[8]), as.numeric(splittedPerformanceMetricValue[10]), 
               as.numeric(splittedPerformanceMetricValue[12]), as.numeric(splittedPerformanceMetricValue[14])); 
      
      rawPerformanceMetricDataFrameStage0Temp <- rawPerformanceMetricDataFrameStage0                   
      rawPerformanceMetricDataFrameStage0Temp[nrow(rawPerformanceMetricDataFrameStage0) + 1, ] <- row
      rawPerformanceMetricDataFrameStage0 <- rawPerformanceMetricDataFrameStage0Temp
    }
  }
}

rawPerformanceMetricDataFrameStage0Temp <- NULL
rowTimeValue <- NULL
rawPerformanceMetricValue <- NULL
splittedPerformanceMetricValue <- NULL


#-------------- Stage 1
rawPerformanceMetricDataFrameStage1 <- NULL
rawPerformanceMetricDataFrameStage1Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage1)) {
  rowTimeValue <- rawEmotivTrackingDataStage1$time[i]
  rawPerformanceMetricValue <- rawEmotivTrackingDataStage1$PerformanceMetricsDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rawPerformanceMetricValue) && !is.na(rawPerformanceMetricValue) && !is.nan(rawPerformanceMetricValue) && length(rawPerformanceMetricValue) && rawPerformanceMetricValue != "")
  {
    splittedPerformanceMetricValue <- stringr::str_split(rawPerformanceMetricValue, ";")[[1]]
    splittedPerformanceMetricValue <- stringr::str_replace(splittedPerformanceMetricValue, ",", ".")
    
    if (is.null(rawPerformanceMetricDataFrameStage1))
    {
      rawPerformanceMetricDataFrameStage1 <- data.frame("time" = c(as.numeric(rowTimeValue)), "eng" = c(as.numeric(rawPerformanceMetricValue[3])), "exc" = c(as.numeric(rawPerformanceMetricValue[5])), 
                                             "lex" = c(as.numeric(rawPerformanceMetricValue[6])), "str" = c(as.numeric(rawPerformanceMetricValue[8])), "rel" = c(as.numeric(rawPerformanceMetricValue[10])), 
                                             "int" = c(as.numeric(rawPerformanceMetricValue[12])), "foc" = c(as.numeric(rawPerformanceMetricValue[14])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(rawPerformanceMetricValue[3]), as.numeric(rawPerformanceMetricValue[5]), 
               as.numeric(rawPerformanceMetricValue[6]),  as.numeric(rawPerformanceMetricValue[8]), as.numeric(rawPerformanceMetricValue[10]), 
               as.numeric(rawPerformanceMetricValue[12]), as.numeric(rawPerformanceMetricValue[14])); 
      
      rawPerformanceMetricDataFrameStage1Temp <- rawPerformanceMetricDataFrameStage1                   
      rawPerformanceMetricDataFrameStage1Temp[nrow(rawPerformanceMetricDataFrameStage1) + 1, ] <- row
      rawPerformanceMetricDataFrameStage1 <- rawPerformanceMetricDataFrameStage1Temp
    }
  }
}

rawPerformanceMetricDataFrameStage1Temp <- NULL
rowTimeValue <- NULL
rawPerformanceMetricValue <- NULL
splittedPerformanceMetricValue <- NULL


#-------------- Stage 2
rawPerformanceMetricDataFrameStage2 <- NULL
rawPerformanceMetricDataFrameStage2Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage2)) {
  rowTimeValue <- rawEmotivTrackingDataStage2$time[i]
  rawPerformanceMetricValue <- rawEmotivTrackingDataStage2$PerformanceMetricsDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rawPerformanceMetricValue) && !is.na(rawPerformanceMetricValue) && !is.nan(rawPerformanceMetricValue) && length(rawPerformanceMetricValue) && rawPerformanceMetricValue != "")
  {
    splittedPerformanceMetricValue <- stringr::str_split(rawPerformanceMetricValue, ";")[[1]]
    splittedPerformanceMetricValue <- stringr::str_replace(splittedPerformanceMetricValue, ",", ".")
    
    if (is.null(rawPerformanceMetricDataFrameStage2))
    {
      rawPerformanceMetricDataFrameStage2 <- tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "eng" = c(as.numeric(splittedPerformanceMetricValue[3])), "exc" = c(as.numeric(splittedPerformanceMetricValue[5])), 
                                  "lex" = c(as.numeric(splittedPerformanceMetricValue[6])), "str" = c(as.numeric(splittedPerformanceMetricValue[8])), "rel" = c(as.numeric(splittedPerformanceMetricValue[10])), 
                                  "int" = c(as.numeric(splittedPerformanceMetricValue[12])), "foc" = c(as.numeric(splittedPerformanceMetricValue[14])));
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(splittedPerformanceMetricValue[3]), as.numeric(splittedPerformanceMetricValue[5]), 
               as.numeric(splittedPerformanceMetricValue[6]),  as.numeric(splittedPerformanceMetricValue[8]), as.numeric(splittedPerformanceMetricValue[10]), 
               as.numeric(splittedPerformanceMetricValue[12]), as.numeric(splittedPerformanceMetricValue[14])); 
      
      rawPerformanceMetricDataFrameStage2Temp <- rawPerformanceMetricDataFrameStage2                   
      rawPerformanceMetricDataFrameStage2Temp[nrow(rawPerformanceMetricDataFrameStage2) + 1, ] <- row
      rawPerformanceMetricDataFrameStage2 <- rawPerformanceMetricDataFrameStage2Temp
      #rawPerformanceMetricDataFrameStage2 <- rbind(rawPerformanceMetricDataFrameStage2, tempDataFrame)
    }
  }
}

rawPerformanceMetricDataFrameStage2Temp <- NULL
rowTimeValue <- NULL
rawPerformanceMetricValue <- NULL
splittedPerformanceMetricValue <- NULL

#plot(rawPerformanceMetricDataFrameStage0$time, rawPerformanceMetricDataFrameStage0$eng, type='l')

