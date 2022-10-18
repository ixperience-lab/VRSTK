# Simple Version of downsampling
# Downsampling to: the samplingrate of ecg and eda 
# Downsampling of: Eye-Tracking, Band-Power, Performance-Metric


downsampling <- function(automationStage, stage)
{
  if(automationStage == 2)
  {
    countSiemensSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countSiemensSamples <- nrow(transformedBitalinoEDASiemensDataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage0)
      tempResultDataFrame <- transformedBitalinoEDASiemensDataFrameStage0
      tempDataFrame <- transformedBitalinoECGDataFrameStage0
    }
    if(stage == 1){
      countSiemensSamples <- nrow(transformedBitalinoEDASiemensDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- transformedBitalinoEDASiemensDataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countSiemensSamples <- nrow(transformedBitalinoEDASiemensDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- transformedBitalinoEDASiemensDataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
    }
    
    print(countSiemensSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countSiemensSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledSiemensDataFrame <- NULL
    for(i in 1:countSiemensSamples) {
      if (samplesCounter == factor && is.null(downsampledSiemensDataFrame)) {
        downsampledSiemensDataFrame <- data.frame("RawValueInMicroSiemens"       = c(as.numeric(tempResultDataFrame[i, 1])), 
                                                  "FilteredValueInMicroSiemens"  = c(as.numeric(tempResultDataFrame[i, 2])));
        
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row <- c(as.numeric(tempResultDataFrame[i, 1]), 
                 as.numeric(tempResultDataFrame[i, 2]))
        
        downsampledSiemensTemp <- downsampledSiemensDataFrame                   
        downsampledSiemensTemp[nrow(downsampledSiemensDataFrame) + 1, ] <- row
        downsampledSiemensDataFrame <- downsampledSiemensTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    
    countSiemensSamples <- nrow(downsampledSiemensDataFrame)
    countTransformedBitalinoSamples <- nrow(tempDataFrame)
    print("results_lengths: ")
    print(countSiemensSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    if (countSiemensSamples > countTransformedBitalinoSamples){
      downsampledSiemensDataFrame$time[1] <- 0
      downsampledSiemensDataFrame$time[2:countSiemensSamples] <-tempDataFrame$time  
    } 
     
    if (countSiemensSamples < countTransformedBitalinoSamples){
      downsampledSiemensDataFrame$time <-tempDataFrame$time[2:countTransformedBitalinoSamples]  
    } 
    
    if (countSiemensSamples == countTransformedBitalinoSamples){
      downsampledSiemensDataFrame$time <-tempDataFrame$time
    }
    
    downsampledSiemensDataFrame$time <- as.integer(downsampledSiemensDataFrame$time)
    downsampledSiemensDataFrame <- downsampledSiemensDataFrame[, c(3, 1, 2)]
    return(downsampledSiemensDataFrame)
  }
  
  
  if(automationStage == 3)
  {
    countBandPowerSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage0)
      tempResultDataFrame <- rawBandPowerDataFrameStage0
      tempDataFrame <- transformedBitalinoECGDataFrameStage0
    }
    if(stage == 1){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawBandPowerDataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countBandPowerSamples <- nrow(rawBandPowerDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- rawBandPowerDataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
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
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage0)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage0
      tempDataFrame <- transformedBitalinoECGDataFrameStage0
    }
    if(stage == 1){
      countPerformanceMetricSamples <- nrow(rawPerformanceMetricDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countPerformanceMetricSamples <- nrow(rawPerformanceMetricDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- rawPerformanceMetricDataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
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
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawQualityParametersPages28114DataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countPageQualityParmetersSamples <- nrow(rawQualityParametersPages28114DataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- rawQualityParametersPages28114DataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
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
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage0
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 1){
      countFixationSaccadsSamples <- nrow(rawEyeTrackingInformationDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countFixationSaccadsSamples <- nrow(rawEyeTrackingInformationDataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- rawEyeTrackingInformationDataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
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
  
  
  if(automationStage == 7)
  {
    countCenterOfViewSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 1){
      countCenterOfViewSamples <- nrow(centerOfViewInformationDataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- centerOfViewInformationDataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    
    print(countCenterOfViewSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countCenterOfViewSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledCenterOfViewDataFrame <- NULL
    for(i in 1:countCenterOfViewSamples) {
      if (samplesCounter == factor && is.null(downsampledCenterOfViewDataFrame)) {
        downsampledCenterOfViewDataFrame <- data.frame("time"                = c(as.integer(tempResultDataFrame[i, 1])), 
                                                       "ActivatedModelIndex" = c(as.numeric(tempResultDataFrame[i, 2])));
        
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-tempResultDataFrame[i,]
        downsampledCenterOfViewTemp <- downsampledCenterOfViewDataFrame                   
        downsampledCenterOfViewTemp[nrow(downsampledCenterOfViewDataFrame) + 1, ] <- row
        downsampledCenterOfViewDataFrame <- downsampledCenterOfViewTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledCenterOfViewDataFrame$time <- as.integer(downsampledCenterOfViewDataFrame$time)
    return(downsampledCenterOfViewDataFrame)
  }
  
  if(automationStage == 8)
  {
    countSaccadesPositionsInfoSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 1){
      countSaccadesPositionsInfoSamples <- nrow(rawEyeTrackingSaccadesPositionsInformationStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- rawEyeTrackingSaccadesPositionsInformationStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    
    print(countSaccadesPositionsInfoSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countSaccadesPositionsInfoSamples %/% countTransformedBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    downsampledSaccadesPositionsInfoDataFrame <- NULL
    for(i in 1:countSaccadesPositionsInfoSamples) {
      if (samplesCounter == factor && is.null(downsampledSaccadesPositionsInfoDataFrame)) {
        downsampledSaccadesPositionsInfoDataFrame <- data.frame("time"      = c(as.integer(tempResultDataFrame[i, 1])), 
                                                                "Saccade0X" = c(as.numeric(tempResultDataFrame[i, 2])), 
                                                                "Saccade0Y" = c(as.numeric(tempResultDataFrame[i, 3])), 
                                                                "Saccade0Z" = c(as.numeric(tempResultDataFrame[i, 4])), 
                                                                "Saccade1X" = c(as.numeric(tempResultDataFrame[i, 5])), 
                                                                "Saccade1Y" = c(as.numeric(tempResultDataFrame[i, 6])), 
                                                                "Saccade1Z" = c(as.numeric(tempResultDataFrame[i, 7])),
                                                                "QuestionId"    = c(as.numeric(tempResultDataFrame[i, 8])),
                                                                "SaccadesDiffX" = c(as.numeric(tempResultDataFrame[i, 9])),
                                                                "SaccadesMeanX" = c(as.numeric(tempResultDataFrame[i, 10])),
                                                                "SaccadesSdX"   = c(as.numeric(tempResultDataFrame[i, 11])),
                                                                "SaccadesMinX"  = c(as.numeric(tempResultDataFrame[i, 12])),
                                                                "SaccadesMaxX"  = c(as.numeric(tempResultDataFrame[i, 13])));
        
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-tempResultDataFrame[i,]
        downsampledSaccadesPositionsInfoTemp <- downsampledSaccadesPositionsInfoDataFrame                   
        downsampledSaccadesPositionsInfoTemp[nrow(downsampledSaccadesPositionsInfoDataFrame) + 1, ] <- row
        downsampledSaccadesPositionsInfoDataFrame <- downsampledSaccadesPositionsInfoTemp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
    downsampledSaccadesPositionsInfoDataFrame$time <- as.integer(downsampledSaccadesPositionsInfoDataFrame$time)
    return(downsampledSaccadesPositionsInfoDataFrame)
  }

}