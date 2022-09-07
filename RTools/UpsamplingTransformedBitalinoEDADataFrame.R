# Upsampling of transformed bitalino EDA DataFrame

upsamplingTransformedBitalinoEdeDataFrame <- function(stage)
{
  #if(automationStage == 3)
  {
    countTransformedEdaBitalinoSamples <- 0
    countTransformedBitalinoSamples <- 1
    tempResultDataFrame <- NULL
    tempDataFrame <- NULL
    if(stage == 0){
      countTransformedEdaBitalinoSamples <- nrow(transformedBitalinoEDADataFrameStage0)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage0)
      tempResultDataFrame <- transformedBitalinoEDADataFrameStage0
      tempDataFrame <- transformedBitalinoECGDataFrameStage0
    }
    if(stage == 1){
      countTransformedEdaBitalinoSamples <- nrow(transformedBitalinoEDADataFrameStage1)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage1)
      tempResultDataFrame <- transformedBitalinoEDADataFrameStage1
      tempDataFrame <- transformedBitalinoECGDataFrameStage1
    }
    if(stage == 2){
      countTransformedEdaBitalinoSamples <- nrow(transformedBitalinoEDADataFrameStage2)
      countTransformedBitalinoSamples <- nrow(transformedBitalinoECGDataFrameStage2)
      tempResultDataFrame <- transformedBitalinoEDADataFrameStage2
      tempDataFrame <- transformedBitalinoECGDataFrameStage2
    }
    
    print(countTransformedEdaBitalinoSamples, zero.print = ".") # quite nicer,
    print(countTransformedBitalinoSamples, zero.print = ".") # quite nicer,
    
    factor <- countTransformedBitalinoSamples - countTransformedEdaBitalinoSamples
    print(factor, zero.print = ".") # quite nicer,
    
    samplesCounter <- 1
    upsampledEdaBitalinoDataFrame <- NULL
    if (factor > 0){
      for(j in 1:countTransformedBitalinoSamples) {
        isSetValue <- FALSE
        ecgTimeInS <- (as.integer(tempDataFrame[j,1]))
        
        for(i in 1:countTransformedEdaBitalinoSamples) {
          sampleTimeInS <- (as.integer(tempResultDataFrame[i,1])%/% 1000)
          
          if (sampleTimeInS ==  ecgTimeInS && is.null(upsampledEdaBitalinoDataFrame)){ 
            upsampledEdaBitalinoDataFrame <- data.frame("time" = c(as.integer(tempDataFrame[j,1])),      
                                                        "onsets" = c(as.numeric(tempResultDataFrame[i,1])), 
                                                        "peaks" = c(as.numeric(tempResultDataFrame[i,2])), 
                                                        "amps" = c(as.numeric(tempResultDataFrame[i,3])));
            isSetValue <- TRUE
            break
          } else if(sampleTimeInS ==  ecgTimeInS && !(is.null(upsampledEdaBitalinoDataFrame))){
            row <- c(as.integer(tempDataFrame[j,1]), 
                     as.numeric(tempResultDataFrame[i,1]), 
                     as.numeric(tempResultDataFrame[i,2]), 
                     as.numeric(tempResultDataFrame[i,3]))
            upsampledTempDataFrame <- upsampledEdaBitalinoDataFrame                   
            upsampledTempDataFrame[nrow(upsampledEdaBitalinoDataFrame) + 1, ] <- row
            upsampledEdaBitalinoDataFrame <- upsampledTempDataFrame
            isSetValue <- TRUE
            break
          }
        }
        
        if (!(isSetValue) && is.null(upsampledEdaBitalinoDataFrame)){ 
          upsampledEdaBitalinoDataFrame <- data.frame("time" = c(as.integer(tempDataFrame[j,1])),      
                                                      "onsets" = c(as.numeric(0)), 
                                                      "peaks" = c(as.numeric(0)), 
                                                      "amps" = c(as.numeric(0)));
        } else if (!(isSetValue) && !(is.null(upsampledEdaBitalinoDataFrame))){
          row <- c(as.integer(tempDataFrame[j,1]), 
                   as.numeric(as.numeric(0)), 
                   as.numeric(as.numeric(0)), 
                   as.numeric(as.numeric(0)))
          upsampledTempDataFrame <- upsampledEdaBitalinoDataFrame                   
          upsampledTempDataFrame[nrow(upsampledEdaBitalinoDataFrame) + 1, ] <- row
          upsampledEdaBitalinoDataFrame <- upsampledTempDataFrame
        }
      }
    }
    upsampledEdaBitalinoDataFrame$time <- as.integer(upsampledEdaBitalinoDataFrame$time)
    return(upsampledEdaBitalinoDataFrame)
  }
}