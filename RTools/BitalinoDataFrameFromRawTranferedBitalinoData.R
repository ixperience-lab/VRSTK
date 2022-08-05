#BitalinoDataFrameFromRawTranferedBitalinoData
#Header
# stream address time - Seq[Sequenz number] : Digital Out[O0 O1 O2 O3] : Analog Channels[A1 A2 A3 A4 A5 A6] 
#Value
# /Bitalino/OpenSinglesStream 1792646.4751661 - Seq[7] : O[false false false false] ; A[6.2744140625 -0.0263671856373549 0 0 0 0]

# A1 = EDA-Value in mOhm
# A2 = ECG-Value in mV

# splitter is " " and ";"

#rawTransferedBitalinoDataFrameStage0 <- data.frame("time" = c(as.numeric(0.0)), "EDA(mOhm)" = c(as.numeric(0.0)), "ECG(mV)" = c(as.numeric(0.0)));

rawTransferedBitalinoDataFrameStage0 <- NULL

for(i in 1:nrow(rawBitalinoDataStage0)) {
  rowTimeValue <- rawBitalinoDataStage0$time[i]
  rowTransferedValue <- rawBitalinoDataStage0$`_transferedReceivedMessage_TrackingBitalinoWithOSC`[i]
  
  if (!is.null(rowTransferedValue) && !is.na(rowTransferedValue) && !is.nan(rowTransferedValue) && length(rowTransferedValue) && rowTransferedValue != "")
  {
    splittedTransferedValue <- stringr::str_split(rowTransferedValue, ";")[[1]]
    splittedTransferedValue <- stringr::str_replace(splittedTransferedValue, "\\[", " ")
    splittedTransferedValue <- stringr::str_replace(splittedTransferedValue, "\\]", " ")
    
    splittedSecondValue <- stringr::str_split(splittedTransferedValue, " ")[[2]]
    
    tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "EDA in mOhm" = c(as.numeric(splittedSecondValue[3])), "ECG in mV" = c(as.numeric(splittedSecondValue[4])));
    
    if (is.null(rawTransferedBitalinoDataFrameStage0))
    {
      rawTransferedBitalinoDataFrameStage0 <- tempDataFrame
    }
    else
    {
      rawTransferedBitalinoDataFrameStage0 <- rbind(rawTransferedBitalinoDataFrameStage0, tempDataFrame)
    }
  }
}

