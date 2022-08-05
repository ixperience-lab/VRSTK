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

#rawPerformanceMetricDataFrameStage0 <- data.frame("time" = c(as.numeric(0.0)), "eng" = c(as.numeric(0.0)), "exc" = c(as.numeric(0.0)), 
#                                                  "lex" = c(as.numeric(0.0)), "str" = c(as.numeric(0.0)), "rel" = c(as.numeric(0.0)), 
#                                                  "int" = c(as.numeric(0.0)), "foc" = c(as.numeric(0.0)));

rawPerformanceMetricDataFrameStage0 <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage0)) {
  rowTimeValue <- rawEmotivTrackingDataStage0$time[i]
  rawPerformanceMetricValue <- rawEmotivTrackingDataStage0$PerformanceMetricsDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rawPerformanceMetricValue) && !is.na(rawPerformanceMetricValue) && !is.nan(rawPerformanceMetricValue) && length(rawPerformanceMetricValue) && rawPerformanceMetricValue != "")
  {
    splittedPerformanceMetricValue <- stringr::str_split(rawPerformanceMetricValue, ";")[[1]]
    splittedPerformanceMetricValue <- stringr::str_replace(splittedPerformanceMetricValue, ",", ".")
    
    tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "eng" = c(as.numeric(splittedPerformanceMetricValue[3])), "exc" = c(as.numeric(splittedBandPowerValues[5])), 
                                "lex" = c(as.numeric(splittedBandPowerValues[6])), "str" = c(as.numeric(splittedBandPowerValues[8])), "rel" = c(as.numeric(splittedBandPowerValues[10])), 
                                "int" = c(as.numeric(splittedBandPowerValues[12])), "foc" = c(as.numeric(splittedBandPowerValues[14])));
    
    if (is.null(rawPerformanceMetricDataFrameStage0))
    {
      rawPerformanceMetricDataFrameStage0 <- tempDataFrame
    }
    else
    {
      rawPerformanceMetricDataFrameStage0 <- rbind(rawPerformanceMetricDataFrameStage0, tempDataFrame)
    }
  }
}