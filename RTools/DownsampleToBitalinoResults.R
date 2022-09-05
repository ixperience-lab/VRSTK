# Simple Version of downsampling
# Downsampling to: the samplingrate of ecg and eda 
# Downsampling of: Eye-Tracking, Band-Power, Performance-Metric


downsampling <- function(automationStage, stage)
{
  if(automationStage == 3)
  {
    countBandPowerSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage0)
      tempResultDataFrame <- rawBandPowerDataFrameStage0
      tempDataFrame <- transformedBilinoECGDataFrameStage0
    }
    if(stage == 1){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage1)
      tempResultDataFrame <- rawBandPowerDataFrameStage1
      tempDataFrame <- transformedBilinoECGDataFrameStage1
    }
    if(stage == 2){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage2)
      tempResultDataFrame <- rawBandPowerDataFrameStage2
      tempDataFrame <- transformedBilinoECGDataFrameStage2
    }
    
    print(countBandPowerSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countBandPowerSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledBandPowerDataFrame <- NULL
    for(i in 1:countBandPowerSamples) {
      if (samplesCounter == factor && is.null(downsampledBandPowerDataFrame)) {
        downsampledBandPowerDataFrame <- data.frame("time" = c(as.integer(tempResultDataFrame[i,1])),      "AF3/theta" = c(as.numeric(tempResultDataFrame[i,2])), 
                                    "AF3/alpha" = c(as.numeric(tempResultDataFrame[i,3])), "AF3/betaL" = c(as.numeric(tempResultDataFrame[i,4])), 
                                    "AF3/betaH" = c(as.numeric(tempResultDataFrame[i,5])), "AF3/gamma" = c(as.numeric(tempResultDataFrame[i,6])),
                                    "F7/theta"  = c(as.numeric(tempResultDataFrame[i,7])), "F7/alpha"  = c(as.numeric(tempResultDataFrame[i,8])),
                                    "F7/betaL"  = c(as.numeric(tempResultDataFrame[i,9])), "F7/betaH"  = c(as.numeric(tempResultDataFrame[i,10])),
                                    "F7/gamma"  = c(as.numeric(tempResultDataFrame[i,11])),
                                    "F3/theta"  = c(as.numeric(tempResultDataFrame[i,12])), "F3/alpha" = c(as.numeric(tempResultDataFrame[i,13])),
                                    "F3/betaL"  = c(as.numeric(tempResultDataFrame[i,14])), "F3/betaH" = c(as.numeric(tempResultDataFrame[i,15])),
                                    "F3/gamma"  = c(as.numeric(tempResultDataFrame[i,16])),
                                    "FC5/theta" = c(as.numeric(tempResultDataFrame[i,17])), "FC5/alpha" = c(as.numeric(tempResultDataFrame[i,18])),
                                    "FC5/betaL" = c(as.numeric(tempResultDataFrame[i,19])), "FC5/betaH" = c(as.numeric(tempResultDataFrame[i,20])),
                                    "FC5/gamma" = c(as.numeric(tempResultDataFrame[i,21])),
                                    "T7/theta"  = c(as.numeric(tempResultDataFrame[i,22])), "T7/alpha" = c(as.numeric(tempResultDataFrame[i,23])),
                                    "T7/betaL"  = c(as.numeric(tempResultDataFrame[i,24])), "T7/betaH" = c(as.numeric(tempResultDataFrame[i,25])),
                                    "T7/gamma"  = c(as.numeric(tempResultDataFrame[i,26])),
                                    "P7/theta"  = c(as.numeric(tempResultDataFrame[i,27])), "P7/alpha" = c(as.numeric(tempResultDataFrame[i,28])),
                                    "P7/betaL"  = c(as.numeric(tempResultDataFrame[i,29])), "P7/betaH" = c(as.numeric(tempResultDataFrame[i,30])),
                                    "P7/gamma"  = c(as.numeric(tempResultDataFrame[i,31])),
                                    "O1/theta"  = c(as.numeric(tempResultDataFrame[i,32])), "O1/alpha" = c(as.numeric(tempResultDataFrame[i,33])),
                                    "O1/betaL"  = c(as.numeric(tempResultDataFrame[i,34])), "O1/betaH" = c(as.numeric(tempResultDataFrame[i,35])),
                                    "O1/gamma"  = c(as.numeric(tempResultDataFrame[i,36])),
                                    "O2/theta"  = c(as.numeric(tempResultDataFrame[i,37])), "O2/alpha" = c(as.numeric(tempResultDataFrame[i,38])),
                                    "O2/betaL"  = c(as.numeric(tempResultDataFrame[i,39])), "O2/betaH" = c(as.numeric(tempResultDataFrame[i,40])),
                                    "O2/gamma"  = c(as.numeric(tempResultDataFrame[i,41])),
                                    "P8/theta"  = c(as.numeric(tempResultDataFrame[i,42])), "P8/alpha" = c(as.numeric(tempResultDataFrame[i,43])),
                                    "P8/betaL"  = c(as.numeric(tempResultDataFrame[i,44])), "P8/betaH" = c(as.numeric(tempResultDataFrame[i,45])),
                                    "P8/gamma"  = c(as.numeric(tempResultDataFrame[i,46])),
                                    "T8/theta"  = c(as.numeric(tempResultDataFrame[i,47])), "T8/alpha" = c(as.numeric(tempResultDataFrame[i,48])),
                                    "T8/betaL"  = c(as.numeric(tempResultDataFrame[i,49])), "T8/betaH" = c(as.numeric(tempResultDataFrame[i,50])),
                                    "T8/gamma"  = c(as.numeric(tempResultDataFrame[i,51])),
                                    "FC6/theta" = c(as.numeric(tempResultDataFrame[i,52])), "FC6/alpha" = c(as.numeric(tempResultDataFrame[i,53])),
                                    "FC6/betaL" = c(as.numeric(tempResultDataFrame[i,54])), "FC6/betaH" = c(as.numeric(tempResultDataFrame[i,55])),
                                    "FC6/gamma" = c(as.numeric(tempResultDataFrame[i,56])),
                                    "F4/theta"  = c(as.numeric(tempResultDataFrame[i,57])), "F4/alpha" = c(as.numeric(tempResultDataFrame[i,58])),
                                    "F4/betaL"  = c(as.numeric(tempResultDataFrame[i,59])), "F4/betaH" = c(as.numeric(tempResultDataFrame[i,60])),
                                    "F4/gamma"  = c(as.numeric(tempResultDataFrame[i,61])),
                                    "F8/theta"  = c(as.numeric(tempResultDataFrame[i,62])), "F8/alpha" = c(as.numeric(tempResultDataFrame[i,63])),
                                    "F8/betaL"  = c(as.numeric(tempResultDataFrame[i,64])), "F8/betaH" = c(as.numeric(tempResultDataFrame[i,65])),
                                    "F8/gamma"  = c(as.numeric(tempResultDataFrame[i,66])),
                                    "AF4/theta" = c(as.numeric(tempResultDataFrame[i,67])), "AF4/alpha" = c(as.numeric(tempResultDataFrame[i,68])),
                                    "AF4/betaL" = c(as.numeric(tempResultDataFrame[i,69])), "AF4/betaH" = c(as.numeric(tempResultDataFrame[i,70])),
                                    "AF4/gamma" = c(as.numeric(tempResultDataFrame[i,71])));
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row <- tempResultDataFrame[i,]
        downsampledTempDataFrame <- downsampledBandPowerDataFrame                   
        downsampledTempDataFrame[nrow(downsampledBandPowerDataFrame) + 1, ] <- row
        downsampledBandPowerDataFrame <- downsampledTempDataFrame
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledBandPowerDataFrame$time <- as.integer(downsampledBandPowerDataFrame$time)
    return(downsampledBandPowerDataFrame)
  }
  
  if(automationStage == 4)
  {
    countPerformanceMetricSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countPerformanceMetricSamples <- nrow(rawPerformanceMetricDataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage0)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage0
      tempDataFrame <- transformedBilinoECGDataFrameStage0
    }
    if(stage == 1){
      countPerformanceMetricSamples <- nrow(rawPerformanceMetricDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage1)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage1
      tempDataFrame <- transformedBilinoECGDataFrameStage1
    }
    if(stage == 2){
      countPerformanceMetricSamples <- nrow(rawPerformanceMetricDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage2)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage2
      tempDataFrame <- transformedBilinoECGDataFrameStage2
    }
    
    print(countPerformanceMetricSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countPerformanceMetricSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledBandPowerDataFrame <- NULL
    for(i in 1:countPerformanceMetricSamples) {
      if (samplesCounter == factor && is.null(downsampledBandPowerDataFrame)) {
        downsampledBandPowerDataFrame <- data.frame("time" = c(as.integer(tempResultDataFrame[i, 1])), 
                                                    "eng"  = c(as.numeric(tempResultDataFrame[i, 2])), 
                                                    "exc"  = c(as.numeric(tempResultDataFrame[i, 3])), 
                                                    "lex"  = c(as.numeric(tempResultDataFrame[i, 4])), 
                                                    "str"  = c(as.numeric(tempResultDataFrame[i, 5])), 
                                                    "rel"  = c(as.numeric(tempResultDataFrame[i, 6])), 
                                                    "int"  = c(as.numeric(tempResultDataFrame[i, 7])), 
                                                    "foc"  = c(as.numeric(tempResultDataFrame[i, 8])));

        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-tempResultDataFrame[i,]
        downsampledPerformanceMetricTemp <- downsampledBandPowerDataFrame                   
        downsampledPerformanceMetricTemp[nrow(downsampledBandPowerDataFrame) + 1, ] <- row
        downsampledBandPowerDataFrame <- downsampledPerformanceMetricTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledBandPowerDataFrame$time <- as.integer(downsampledBandPowerDataFrame$time)
    return(downsampledBandPowerDataFrame)
  }
  
  if(automationStage == 5)
  {
    countPageQualityParmetersSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 1){
      countPageQualityParmetersSamples <- nrow(rawQualityParametersPages28114DataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage1)
      tempResultDataFrame <- rawQualityParametersPages28114DataFrameStage1
      tempDataFrame <- transformedBilinoECGDataFrameStage1
    }
    if(stage == 2){
      countPageQualityParmetersSamples <- nrow(rawQualityParametersPages28114DataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage2)
      tempResultDataFrame <- rawQualityParametersPages28114DataFrameStage2
      tempDataFrame <- transformedBilinoECGDataFrameStage2
    }
    
    print(countPageQualityParmetersSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countPageQualityParmetersSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    if (factor == 1) factor = 2
      
    samplesCounter <- 1
    downsampledPageQualityParmetersDataFrame <- NULL
    for(i in 1:countPageQualityParmetersSamples) {
      if (samplesCounter == factor && is.null(downsampledPageQualityParmetersDataFrame)) {
        downsampledPageQualityParmetersDataFrame <- data.frame("time"     = c(as.integer(tempResultDataFrame[i, 1])), 
                                                               "STARTED"  = c(tempResultDataFrame[i, 2]), 
                                                               "LASTDATA" = c(tempResultDataFrame[i, 3]), 
                                                               "LASTPAGE" = c(as.numeric(tempResultDataFrame[i, 4])), 
                                                               "MAXPAGE"  = c(as.numeric(tempResultDataFrame[i, 5])), 
                                                               "FINISHED" = c(as.numeric(tempResultDataFrame[i, 6])), 
                                                               "TIME_SUM" = c(as.numeric(tempResultDataFrame[i, 7])), 
                                                               "TIME_RSI" = c(as.numeric(tempResultDataFrame[i, 8])), 
                                                               "MISSING"  = c(as.numeric(tempResultDataFrame[i, 9])), 
                                                               "DEG_TIME" = c(as.numeric(tempResultDataFrame[i, 10])), 
                                                               "DegTimeThreshold"  = c(as.numeric(tempResultDataFrame[i, 11])), 
                                                               "DegTimeLowQuality" = c(tempResultDataFrame[i, 12]), 
                                                               "DegTimeThresholdForOnePage" = c(as.numeric(tempResultDataFrame[i, 13])),
                                                               "DegTimeValueForOnePage"     = c(as.numeric(tempResultDataFrame[i, 14])), 
                                                               "CurrentPageNumber"          = c(as.numeric(tempResultDataFrame[i, 15])),
                                                               "StandardDeviationStraightLineAnswer" = c(as.numeric(tempResultDataFrame[i, 16])), 
                                                               "AbsoluteDerivationOfResponseValue"   = c(as.numeric(tempResultDataFrame[i, 17])));
        
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-tempResultDataFrame[i,]
        downsampledPageQualityParmetersTemp <- downsampledPageQualityParmetersDataFrame                   
        downsampledPageQualityParmetersTemp[nrow(downsampledPageQualityParmetersDataFrame) + 1, ] <- row
        downsampledPageQualityParmetersDataFrame <- downsampledPageQualityParmetersTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledPageQualityParmetersDataFrame$time <- as.integer(downsampledPageQualityParmetersDataFrame$time)
    return(downsampledPageQualityParmetersDataFrame)
  }
  
  if(automationStage == 6)
  {
    countFixationSaccadsSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countFixationSaccadsSamples <- nrow(rawEyeTrackingInformationDataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage1)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage0
      tempDataFrame <- transformedBilinoECGDataFrameStage1
    }
    if(stage == 1){
      countFixationSaccadsSamples <- nrow(rawEyeTrackingInformationDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage1)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage1
      tempDataFrame <- transformedBilinoECGDataFrameStage1
    }
    if(stage == 2){
      countFixationSaccadsSamples <- nrow(rawEyeTrackingInformationDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage2)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage2
      tempDataFrame <- transformedBilinoECGDataFrameStage2
    }
    
    print(countFixationSaccadsSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countFixationSaccadsSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledFixationSaccadsDataFrame <- NULL
    for(i in 1:countFixationSaccadsSamples) {
      if (samplesCounter == factor && is.null(downsampledFixationSaccadsDataFrame)) {
        downsampledFixationSaccadsDataFrame <- data.frame("time"                  = c(as.integer(tempResultDataFrame[i, 1])), 
                                                          "LeftEyeOpenness"       = c(as.numeric(tempResultDataFrame[i, 2])), 
                                                          "LeftPupilDiameter"     = c(as.numeric(tempResultDataFrame[i, 3])), 
                                                          "RightEyeOpenness"      = c(as.numeric(tempResultDataFrame[i, 4])), 
                                                          "RightPupilDiameter"    = c(as.numeric(tempResultDataFrame[i, 5])), 
                                                          "TotalFixationCounter"  = c(as.numeric(tempResultDataFrame[i, 6])), 
                                                          "FixationCounter"       = c(as.numeric(tempResultDataFrame[i, 7])), 
                                                          "TotalFixationDuration" = c(as.numeric(tempResultDataFrame[i, 8])),
                                                          "FixationDuration"      = c(as.numeric(tempResultDataFrame[i, 9])),
                                                          "MeasuredVelocity"      = c(as.numeric(tempResultDataFrame[i, 10])),
                                                          "SaccadeCounter"        = c(as.numeric(tempResultDataFrame[i, 11])));
        
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-tempResultDataFrame[i,]
        downsampledFixationSaccadsTemp <- downsampledFixationSaccadsDataFrame                   
        downsampledFixationSaccadsTemp[nrow(downsampledFixationSaccadsDataFrame) + 1, ] <- row
        downsampledFixationSaccadsDataFrame <- downsampledFixationSaccadsTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledFixationSaccadsDataFrame$time <- as.integer(downsampledFixationSaccadsDataFrame$time)
    return(downsampledFixationSaccadsDataFrame)
  }

}