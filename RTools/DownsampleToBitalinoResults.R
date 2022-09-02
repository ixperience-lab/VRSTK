# Simple Version of downsampling
# Downsampling to: the samplingrate of ecg and eda 
# Downsampling of: Eye-Tracking, Band-Power, Performance-Metric


downsampling <- function(automationStage)
{
  if(automationStage == 3)
  {
    countBandPowerSamples <- nrow(rawBandPowerDataFrameStage0)
    countTransformedBitalinoSamples <- nrow(transformedBilinoECGDataFrameStage0)
    
    factor <- countBandPowerSamples %/% countTransformedBitalinoSamples
    samplesCounter <- 1
    downsampledBandPowerDataFrameStage0 <- NULL
    for(i in 1:nrow(rawBandPowerDataFrameStage0)) {
      if (samplesCounter == factor && is.null(downsampledBandPowerDataFrameStage0)) {
        tempDataFrame <- data.frame("time" = c(as.numeric(rawBandPowerDataFrameStage0[i,1])), "AF3/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,2])), 
                                    "AF3/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,3])), "AF3/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,4])), 
                                    "AF3/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,5])), "AF3/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,6])),
                                    "F7/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,7])), "F7/alpha"   = c(as.numeric(rawBandPowerDataFrameStage0[i,8])),
                                    "F7/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,9])), "F7/betaH"   = c(as.numeric(rawBandPowerDataFrameStage0[i,10])),
                                    "F7/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,11])),
                                    "F3/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,12])), "F3/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,13])),
                                    "F3/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,14])), "F3/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,15])),
                                    "F3/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,16])),
                                    "FC5/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,17])), "FC5/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,18])),
                                    "FC5/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,19])), "FC5/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,20])),
                                    "FC5/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,21])),
                                    "T7/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,22])), "T7/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,23])),
                                    "T7/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,24])), "T7/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,25])),
                                    "T7/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,26])),
                                    "P7/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,27])), "P7/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,28])),
                                    "P7/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,29])), "P7/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,30])),
                                    "P7/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,31])),
                                    "O1/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,32])), "O1/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,33])),
                                    "O1/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,34])), "O1/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,35])),
                                    "O1/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,36])),
                                    "O2/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,37])), "O2/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,38])),
                                    "O2/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,39])), "O2/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,40])),
                                    "O2/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,41])),
                                    "P8/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,42])), "P8/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,43])),
                                    "P8/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,44])), "P8/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,45])),
                                    "P8/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,46])),
                                    "T8/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,47])), "T8/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,48])),
                                    "T8/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,49])), "T8/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,50])),
                                    "T8/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,51])),
                                    "FC6/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,52])), "FC6/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,53])),
                                    "FC6/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,54])), "FC6/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,55])),
                                    "FC6/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,56])),
                                    "F4/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,57])), "F4/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,58])),
                                    "F4/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,59])), "F4/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,60])),
                                    "F4/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,61])),
                                    "F8/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,62])), "F8/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,63])),
                                    "F8/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,64])), "F8/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,65])),
                                    "F8/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,66])),
                                    "AF4/theta" = c(as.numeric(rawBandPowerDataFrameStage0[i,67])), "AF4/alpha" = c(as.numeric(rawBandPowerDataFrameStage0[i,68])),
                                    "AF4/betaL" = c(as.numeric(rawBandPowerDataFrameStage0[i,69])), "AF4/betaH" = c(as.numeric(rawBandPowerDataFrameStage0[i,70])),
                                    "AF4/gamma" = c(as.numeric(rawBandPowerDataFrameStage0[i,71])));
        
        downsampledBandPowerDataFrameStage0 <- tempDataFrame
        samplesCounter <- 0
      }
      else if(samplesCounter == factor){
        row<-rawBandPowerDataFrameStage0[i,]
        downsampledBandPowerDataFrameStage0Temp <- downsampledBandPowerDataFrameStage0                   
        downsampledBandPowerDataFrameStage0Temp[nrow(downsampledBandPowerDataFrameStage0) + 1, ] <- row
        downsampledBandPowerDataFrameStage0 <- downsampledBandPowerDataFrameStage0Temp
        samplesCounter <- 0
      } 
      
      if (samplesCounter == 0)
        samplesCounter <- 1
      else
        samplesCounter <- samplesCounter + 1
    }
  }
  
  if(automationStage == 4)
  {
    
  }
  
  if(automationStage == 5)
  {
    
  }
  
  if(automationStage == 6)
  {
    
  }
  
  if(automationStage == 7)
  {
    
  }
  
  if(automationStage == 8)
  {
    
  }
  
  if(automationStage == 9)
  {
    
  }

}