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

#Header
# stream address time - Seq[Sequenz number] : Digital Out[O0 O1 O2 O3] : Analog Channels[A1 A2 A3 A4 A5 A6] 
#Value
# /Bitalino/OpenSinglesStream 1792646.6611172 - Seq[1] : O[false false false false] ; A[257 505 0 0 0 0]

# A1 = raw EDA-Value
# A2 = raw ECG-Value

# splitter is " " and ";"

#rawBitalinoDataFrameStage0 <- data.frame("time" = c(as.numeric(0.0)), "EDA raw" = c(as.numeric(0.0)), "ECG raw" = c(as.numeric(0.0)));

rawBitalinoDataFrameStage0 <- NULL

for(i in 1:nrow(rawBitalinoDataStage0)) {
  rowTimeValue <- rawBitalinoDataStage0$time[i]
  rawBitalinoValue <- rawBitalinoDataStage0$`_rawReceivedMessage_TrackingBitalinoWithOSC`[i]
  
  if (!is.null(rawBitalinoValue) && !is.na(rawBitalinoValue) && !is.nan(rawBitalinoValue) && length(rawBitalinoValue) && rawBitalinoValue != "")
  {
    splittedRawValue <- stringr::str_split(rawBitalinoValue, ";")[[1]]
    splittedRawValue <- stringr::str_replace(splittedRawValue, "\\[", " ")
    splittedRawValue <- stringr::str_replace(splittedRawValue, "\\]", " ")
    
    splittedSecondRawValue <- stringr::str_split(splittedRawValue, " ")[[2]]
    
    tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "EDA raw" = c(as.numeric(splittedSecondRawValue[3])), "ECG raw" = c(as.numeric(splittedSecondRawValue[4])));
    
    if (is.null(rawBitalinoDataFrameStage0))
    {
      rawBitalinoDataFrameStage0 <- tempDataFrame
    }
    else
    {
      rawBitalinoDataFrameStage0 <- rbind(rawBitalinoDataFrameStage0, tempDataFrame)
    }
  }
}

# point plot
#plot(rawTransferedBitalinoDataFrameStage0$time, rawTransferedBitalinoDataFrameStage0$ECG.in.mV)
#plot(rawBitalinoDataFrameStage0$time, rawBitalinoDataFrameStage0$ECG.raw)

# wave plot
#plot(rawTransferedBitalinoDataFrameStage0$time, rawTransferedBitalinoDataFrameStage0$ECG.in.mV, col='orange',type='s')
#plot(rawBitalinoDataFrameStage0$time, rawBitalinoDataFrameStage0$ECG.raw, type='s')
#plot(rawBitalinoDataFrameStage0$time, rawBitalinoDataFrameStage0$ECG.raw, type='l')

#write.csv2(rawBitalinoDataFrameStage0, file = "Bitalino_Raw_data_Stage0.csv")
write.table(rawBitalinoDataFrameStage0, file = "Bitalino_Raw_data_Stage0.txt", sep = " ")
