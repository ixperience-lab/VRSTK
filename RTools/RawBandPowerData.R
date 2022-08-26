#Header
#"AF3/theta","AF3/alpha","AF3/betaL","AF3/betaH","AF3/gamma",
#"F7/theta","F7/alpha","F7/betaL","F7/betaH","F7/gamma",
#"F3/theta","F3/alpha","F3/betaL","F3/betaH","F3/gamma",
#"FC5/theta","FC5/alpha","FC5/betaL","FC5/betaH","FC5/gamma",
#"T7/theta","T7/alpha","T7/betaL","T7/betaH","T7/gamma",
#"P7/theta","P7/alpha","P7/betaL","P7/betaH","P7/gamma",
#"O1/theta","O1/alpha","O1/betaL","O1/betaH","O1/gamma",
#"O2/theta","O2/alpha","O2/betaL","O2/betaH","O2/gamma",
#"P8/theta","P8/alpha","P8/betaL","P8/betaH","P8/gamma",
#"T8/theta","T8/alpha","T8/betaL","T8/betaH","T8/gamma",
#"FC6/theta","FC6/alpha","FC6/betaL","FC6/betaH","FC6/gamma",
#"F4/theta","F4/alpha","F4/betaL","F4/betaH","F4/gamma",
#"F8/theta","F8/alpha","F8/betaL","F8/betaH","F8/gamma",
#"AF4/theta","AF4/alpha","AF4/betaL","AF4/betaH","AF4/gamma"
#Value
#"pow":[
#  0.225,0.213,0.537,0.19,0.34,
#  0.511,0.808,1.706,0.839,0.416,
#  ...
#  0.92,0.469,1.657,1.443,0.912,
#  2.675,0.824,0.951,0.303,0.881
#],
#"sid":"f581b2bb-c043-4a00-8737-1e8e09a9a81b",
#"time":1559902987.133

#pow data: 1659612203,3161;
#5,402;2,635;0,835;0,134;0,21;1,472;1,4;0,633;0,177;0,349;1,334;2,991;0,437;0,121;
#0,218;4,08;2,759;1,174;0,12;0,292;1,174;0,85;0,337;0,133;0,278;0,792;0,345;0,365;
#0,203;0,192;1,722;0,93;1,684;1,038;1,199;2,815;0,71;1,049;0,363;0,715;0,443;1,376;
#0,769;0,841;0,583;0,696;2,505;0,967;0,548;0,51;2,422;3,026;1,249;0,24;0,65;2,288;
#3,347;0,956;0,227;0,515;0,961;2,536;1,155;0,247;0,579;3,963;3,518;1,003;0,137;0,297;

# splitter is ";"

# ------------- Tage 0
rawBandPowerDataFrameStage0 <- NULL
rawBandPowerDataFrameStage0Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage0)) {
  rowTimeValue <- rawEmotivTrackingDataStage0$time[i]
  rowBandPowerValue <- rawEmotivTrackingDataStage0$BandPowerDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rowBandPowerValue) && !is.na(rowBandPowerValue) && !is.nan(rowBandPowerValue) && length(rowBandPowerValue) && rowBandPowerValue != "")
  {
    splittedBandPowerValues <- stringr::str_split(rowBandPowerValue, ";")[[1]]
    splittedBandPowerValues <- stringr::str_replace(splittedBandPowerValues, ",", ".")
    
    if (is.null(rawBandPowerDataFrameStage0))
    {
      tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "AF3/theta" = c(as.numeric(splittedBandPowerValues[2])), "AF3/alpha" = c(as.numeric(splittedBandPowerValues[3])), "AF3/betaL" = c(as.numeric(splittedBandPowerValues[4])), "AF3/betaH" = c(as.numeric(splittedBandPowerValues[5])), "AF3/gamma" = c(as.numeric(splittedBandPowerValues[6])),
                                  "F7/theta" = c(as.numeric(splittedBandPowerValues[7])), "F7/alpha" = c(as.numeric(splittedBandPowerValues[8])),"F7/betaL" = c(as.numeric(splittedBandPowerValues[9])),"F7/betaH" = c(as.numeric(splittedBandPowerValues[10])),"F7/gamma" = c(as.numeric(splittedBandPowerValues[11])),
                                  "F3/theta" = c(as.numeric(splittedBandPowerValues[12])), "F3/alpha" = c(as.numeric(splittedBandPowerValues[13])),"F3/betaL" = c(as.numeric(splittedBandPowerValues[14])),"F3/betaH" = c(as.numeric(splittedBandPowerValues[15])),"F3/gamma" = c(as.numeric(splittedBandPowerValues[16])),
                                  "FC5/theta" = c(as.numeric(splittedBandPowerValues[17])),"FC5/alpha" = c(as.numeric(splittedBandPowerValues[18])),"FC5/betaL" = c(as.numeric(splittedBandPowerValues[19])),"FC5/betaH" = c(as.numeric(splittedBandPowerValues[20])),"FC5/gamma" = c(as.numeric(splittedBandPowerValues[21])),
                                  "T7/theta" = c(as.numeric(splittedBandPowerValues[22])), "T7/alpha" = c(as.numeric(splittedBandPowerValues[23])),"T7/betaL" = c(as.numeric(splittedBandPowerValues[24])),"T7/betaH" = c(as.numeric(splittedBandPowerValues[25])),"T7/gamma" = c(as.numeric(splittedBandPowerValues[26])),
                                  "P7/theta" = c(as.numeric(splittedBandPowerValues[27])), "P7/alpha" = c(as.numeric(splittedBandPowerValues[28])),"P7/betaL" = c(as.numeric(splittedBandPowerValues[29])),"P7/betaH" = c(as.numeric(splittedBandPowerValues[30])),"P7/gamma" = c(as.numeric(splittedBandPowerValues[31])),
                                  "O1/theta" = c(as.numeric(splittedBandPowerValues[32])), "O1/alpha" = c(as.numeric(splittedBandPowerValues[33])),"O1/betaL" = c(as.numeric(splittedBandPowerValues[34])),"O1/betaH" = c(as.numeric(splittedBandPowerValues[35])),"O1/gamma" = c(as.numeric(splittedBandPowerValues[36])),
                                  "O2/theta" = c(as.numeric(splittedBandPowerValues[37])), "O2/alpha" = c(as.numeric(splittedBandPowerValues[38])),"O2/betaL" = c(as.numeric(splittedBandPowerValues[39])),"O2/betaH" = c(as.numeric(splittedBandPowerValues[40])),"O2/gamma" = c(as.numeric(splittedBandPowerValues[41])),
                                  "P8/theta" = c(as.numeric(splittedBandPowerValues[42])), "P8/alpha" = c(as.numeric(splittedBandPowerValues[43])),"P8/betaL" = c(as.numeric(splittedBandPowerValues[44])),"P8/betaH" = c(as.numeric(splittedBandPowerValues[45])),"P8/gamma" = c(as.numeric(splittedBandPowerValues[46])),
                                  "T8/theta" = c(as.numeric(splittedBandPowerValues[47])), "T8/alpha" = c(as.numeric(splittedBandPowerValues[48])),"T8/betaL" = c(as.numeric(splittedBandPowerValues[49])),"T8/betaH" = c(as.numeric(splittedBandPowerValues[50])),"T8/gamma" = c(as.numeric(splittedBandPowerValues[51])),
                                  "FC6/theta" = c(as.numeric(splittedBandPowerValues[52])),"FC6/alpha" = c(as.numeric(splittedBandPowerValues[53])),"FC6/betaL" = c(as.numeric(splittedBandPowerValues[54])),"FC6/betaH" = c(as.numeric(splittedBandPowerValues[55])),"FC6/gamma" = c(as.numeric(splittedBandPowerValues[56])),
                                  "F4/theta" = c(as.numeric(splittedBandPowerValues[57])), "F4/alpha" = c(as.numeric(splittedBandPowerValues[58])),"F4/betaL" = c(as.numeric(splittedBandPowerValues[59])),"F4/betaH" = c(as.numeric(splittedBandPowerValues[60])),"F4/gamma" = c(as.numeric(splittedBandPowerValues[61])),
                                  "F8/theta" = c(as.numeric(splittedBandPowerValues[62])), "F8/alpha" = c(as.numeric(splittedBandPowerValues[63])),"F8/betaL" = c(as.numeric(splittedBandPowerValues[64])),"F8/betaH" = c(as.numeric(splittedBandPowerValues[65])),"F8/gamma" = c(as.numeric(splittedBandPowerValues[66])),
                                  "AF4/theta" = c(as.numeric(splittedBandPowerValues[67])),"AF4/alpha" = c(as.numeric(splittedBandPowerValues[68])),"AF4/betaL" = c(as.numeric(splittedBandPowerValues[69])),"AF4/betaH" = c(as.numeric(splittedBandPowerValues[70])),"AF4/gamma" = c(as.numeric(splittedBandPowerValues[71])));
      
      rawBandPowerDataFrameStage0 <- tempDataFrame
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(splittedBandPowerValues[2]), as.numeric(splittedBandPowerValues[3]), as.numeric(splittedBandPowerValues[4]), as.numeric(splittedBandPowerValues[5]), as.numeric(splittedBandPowerValues[6]),
               as.numeric(splittedBandPowerValues[7]),  as.numeric(splittedBandPowerValues[8]),  as.numeric(splittedBandPowerValues[9]),  as.numeric(splittedBandPowerValues[10]), as.numeric(splittedBandPowerValues[11]),
               as.numeric(splittedBandPowerValues[12]), as.numeric(splittedBandPowerValues[13]), as.numeric(splittedBandPowerValues[14]), as.numeric(splittedBandPowerValues[15]), as.numeric(splittedBandPowerValues[16]),
               as.numeric(splittedBandPowerValues[17]), as.numeric(splittedBandPowerValues[18]), as.numeric(splittedBandPowerValues[19]), as.numeric(splittedBandPowerValues[20]), as.numeric(splittedBandPowerValues[21]),
               as.numeric(splittedBandPowerValues[22]), as.numeric(splittedBandPowerValues[23]), as.numeric(splittedBandPowerValues[24]), as.numeric(splittedBandPowerValues[25]), as.numeric(splittedBandPowerValues[26]),
               as.numeric(splittedBandPowerValues[27]), as.numeric(splittedBandPowerValues[28]), as.numeric(splittedBandPowerValues[29]), as.numeric(splittedBandPowerValues[30]), as.numeric(splittedBandPowerValues[31]),
               as.numeric(splittedBandPowerValues[32]), as.numeric(splittedBandPowerValues[33]), as.numeric(splittedBandPowerValues[34]), as.numeric(splittedBandPowerValues[35]), as.numeric(splittedBandPowerValues[36]),
               as.numeric(splittedBandPowerValues[37]), as.numeric(splittedBandPowerValues[38]), as.numeric(splittedBandPowerValues[39]), as.numeric(splittedBandPowerValues[40]), as.numeric(splittedBandPowerValues[41]),
               as.numeric(splittedBandPowerValues[42]), as.numeric(splittedBandPowerValues[43]), as.numeric(splittedBandPowerValues[44]), as.numeric(splittedBandPowerValues[45]), as.numeric(splittedBandPowerValues[46]),
               as.numeric(splittedBandPowerValues[47]), as.numeric(splittedBandPowerValues[48]), as.numeric(splittedBandPowerValues[49]), as.numeric(splittedBandPowerValues[50]), as.numeric(splittedBandPowerValues[51]),
               as.numeric(splittedBandPowerValues[52]), as.numeric(splittedBandPowerValues[53]), as.numeric(splittedBandPowerValues[54]), as.numeric(splittedBandPowerValues[55]), as.numeric(splittedBandPowerValues[56]),
               as.numeric(splittedBandPowerValues[57]), as.numeric(splittedBandPowerValues[58]), as.numeric(splittedBandPowerValues[59]), as.numeric(splittedBandPowerValues[60]), as.numeric(splittedBandPowerValues[61]),
               as.numeric(splittedBandPowerValues[62]), as.numeric(splittedBandPowerValues[63]), as.numeric(splittedBandPowerValues[64]), as.numeric(splittedBandPowerValues[65]), as.numeric(splittedBandPowerValues[66]),
               as.numeric(splittedBandPowerValues[67]), as.numeric(splittedBandPowerValues[68]), as.numeric(splittedBandPowerValues[69]), as.numeric(splittedBandPowerValues[70]), as.numeric(splittedBandPowerValues[71]));
      
      rawBandPowerDataFrameStage0Temp <- rawBandPowerDataFrameStage0                   
      rawBandPowerDataFrameStage0Temp[nrow(rawBandPowerDataFrameStage0) + 1, ] <- row
      rawBandPowerDataFrameStage0 <- rawBandPowerDataFrameStage0Temp
    }
  }
}

rawBandPowerDataFrameStage0Temp <- NULL
row <- NULL
tempDataFrame <- NULL
splittedBandPowerValues <- NULL
rowBandPowerValue <- NULL
rowTimeValue <- NULL

#plot with more then one lines
#plot(rawBandPowerDataFrameStage0$time, rawBandPowerDataFrameStage0$AF3.betaL, type='l')
#lines(rawBandPowerDataFrameStage0$time, rawBandPowerDataFrameStage0$AF3,col='green')
#lines(rawBandPowerDataFrameStage0$time, rawBandPowerDataFrameStage0$F7.betaL,col='blue')

#abline(rawBandPowerDataFrameStage0$AF3.alpha,col="dark green",lty=2)

# ------------- Tage 1
rawBandPowerDataFrameStage1 <- NULL
rawBandPowerDataFrameStage1Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage1)) {
  rowTimeValue <- rawEmotivTrackingDataStage1$time[i]
  rowBandPowerValue <- rawEmotivTrackingDataStage1$BandPowerDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rowBandPowerValue) && !is.na(rowBandPowerValue) && !is.nan(rowBandPowerValue) && length(rowBandPowerValue) && rowBandPowerValue != "")
  {
    splittedBandPowerValues <- stringr::str_split(rowBandPowerValue, ";")[[1]]
    splittedBandPowerValues <- stringr::str_replace(splittedBandPowerValues, ",", ".")
    
    if (is.null(rawBandPowerDataFrameStage1))
    {
      tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "AF3/theta" = c(as.numeric(splittedBandPowerValues[2])), "AF3/alpha" = c(as.numeric(splittedBandPowerValues[3])), "AF3/betaL" = c(as.numeric(splittedBandPowerValues[4])), "AF3/betaH" = c(as.numeric(splittedBandPowerValues[5])), "AF3/gamma" = c(as.numeric(splittedBandPowerValues[6])),
                                  "F7/theta" = c(as.numeric(splittedBandPowerValues[7])), "F7/alpha" = c(as.numeric(splittedBandPowerValues[8])),"F7/betaL" = c(as.numeric(splittedBandPowerValues[9])),"F7/betaH" = c(as.numeric(splittedBandPowerValues[10])),"F7/gamma" = c(as.numeric(splittedBandPowerValues[11])),
                                  "F3/theta" = c(as.numeric(splittedBandPowerValues[12])), "F3/alpha" = c(as.numeric(splittedBandPowerValues[13])),"F3/betaL" = c(as.numeric(splittedBandPowerValues[14])),"F3/betaH" = c(as.numeric(splittedBandPowerValues[15])),"F3/gamma" = c(as.numeric(splittedBandPowerValues[16])),
                                  "FC5/theta" = c(as.numeric(splittedBandPowerValues[17])),"FC5/alpha" = c(as.numeric(splittedBandPowerValues[18])),"FC5/betaL" = c(as.numeric(splittedBandPowerValues[19])),"FC5/betaH" = c(as.numeric(splittedBandPowerValues[20])),"FC5/gamma" = c(as.numeric(splittedBandPowerValues[21])),
                                  "T7/theta" = c(as.numeric(splittedBandPowerValues[22])), "T7/alpha" = c(as.numeric(splittedBandPowerValues[23])),"T7/betaL" = c(as.numeric(splittedBandPowerValues[24])),"T7/betaH" = c(as.numeric(splittedBandPowerValues[25])),"T7/gamma" = c(as.numeric(splittedBandPowerValues[26])),
                                  "P7/theta" = c(as.numeric(splittedBandPowerValues[27])), "P7/alpha" = c(as.numeric(splittedBandPowerValues[28])),"P7/betaL" = c(as.numeric(splittedBandPowerValues[29])),"P7/betaH" = c(as.numeric(splittedBandPowerValues[30])),"P7/gamma" = c(as.numeric(splittedBandPowerValues[31])),
                                  "O1/theta" = c(as.numeric(splittedBandPowerValues[32])), "O1/alpha" = c(as.numeric(splittedBandPowerValues[33])),"O1/betaL" = c(as.numeric(splittedBandPowerValues[34])),"O1/betaH" = c(as.numeric(splittedBandPowerValues[35])),"O1/gamma" = c(as.numeric(splittedBandPowerValues[36])),
                                  "O2/theta" = c(as.numeric(splittedBandPowerValues[37])), "O2/alpha" = c(as.numeric(splittedBandPowerValues[38])),"O2/betaL" = c(as.numeric(splittedBandPowerValues[39])),"O2/betaH" = c(as.numeric(splittedBandPowerValues[40])),"O2/gamma" = c(as.numeric(splittedBandPowerValues[41])),
                                  "P8/theta" = c(as.numeric(splittedBandPowerValues[42])), "P8/alpha" = c(as.numeric(splittedBandPowerValues[43])),"P8/betaL" = c(as.numeric(splittedBandPowerValues[44])),"P8/betaH" = c(as.numeric(splittedBandPowerValues[45])),"P8/gamma" = c(as.numeric(splittedBandPowerValues[46])),
                                  "T8/theta" = c(as.numeric(splittedBandPowerValues[47])), "T8/alpha" = c(as.numeric(splittedBandPowerValues[48])),"T8/betaL" = c(as.numeric(splittedBandPowerValues[49])),"T8/betaH" = c(as.numeric(splittedBandPowerValues[50])),"T8/gamma" = c(as.numeric(splittedBandPowerValues[51])),
                                  "FC6/theta" = c(as.numeric(splittedBandPowerValues[52])),"FC6/alpha" = c(as.numeric(splittedBandPowerValues[53])),"FC6/betaL" = c(as.numeric(splittedBandPowerValues[54])),"FC6/betaH" = c(as.numeric(splittedBandPowerValues[55])),"FC6/gamma" = c(as.numeric(splittedBandPowerValues[56])),
                                  "F4/theta" = c(as.numeric(splittedBandPowerValues[57])), "F4/alpha" = c(as.numeric(splittedBandPowerValues[58])),"F4/betaL" = c(as.numeric(splittedBandPowerValues[59])),"F4/betaH" = c(as.numeric(splittedBandPowerValues[60])),"F4/gamma" = c(as.numeric(splittedBandPowerValues[61])),
                                  "F8/theta" = c(as.numeric(splittedBandPowerValues[62])), "F8/alpha" = c(as.numeric(splittedBandPowerValues[63])),"F8/betaL" = c(as.numeric(splittedBandPowerValues[64])),"F8/betaH" = c(as.numeric(splittedBandPowerValues[65])),"F8/gamma" = c(as.numeric(splittedBandPowerValues[66])),
                                  "AF4/theta" = c(as.numeric(splittedBandPowerValues[67])),"AF4/alpha" = c(as.numeric(splittedBandPowerValues[68])),"AF4/betaL" = c(as.numeric(splittedBandPowerValues[69])),"AF4/betaH" = c(as.numeric(splittedBandPowerValues[70])),"AF4/gamma" = c(as.numeric(splittedBandPowerValues[71])));
      
      rawBandPowerDataFrameStage1 <- tempDataFrame
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(splittedBandPowerValues[2]), as.numeric(splittedBandPowerValues[3]), as.numeric(splittedBandPowerValues[4]), as.numeric(splittedBandPowerValues[5]), as.numeric(splittedBandPowerValues[6]),
               as.numeric(splittedBandPowerValues[7]),  as.numeric(splittedBandPowerValues[8]),  as.numeric(splittedBandPowerValues[9]),  as.numeric(splittedBandPowerValues[10]), as.numeric(splittedBandPowerValues[11]),
               as.numeric(splittedBandPowerValues[12]), as.numeric(splittedBandPowerValues[13]), as.numeric(splittedBandPowerValues[14]), as.numeric(splittedBandPowerValues[15]), as.numeric(splittedBandPowerValues[16]),
               as.numeric(splittedBandPowerValues[17]), as.numeric(splittedBandPowerValues[18]), as.numeric(splittedBandPowerValues[19]), as.numeric(splittedBandPowerValues[20]), as.numeric(splittedBandPowerValues[21]),
               as.numeric(splittedBandPowerValues[22]), as.numeric(splittedBandPowerValues[23]), as.numeric(splittedBandPowerValues[24]), as.numeric(splittedBandPowerValues[25]), as.numeric(splittedBandPowerValues[26]),
               as.numeric(splittedBandPowerValues[27]), as.numeric(splittedBandPowerValues[28]), as.numeric(splittedBandPowerValues[29]), as.numeric(splittedBandPowerValues[30]), as.numeric(splittedBandPowerValues[31]),
               as.numeric(splittedBandPowerValues[32]), as.numeric(splittedBandPowerValues[33]), as.numeric(splittedBandPowerValues[34]), as.numeric(splittedBandPowerValues[35]), as.numeric(splittedBandPowerValues[36]),
               as.numeric(splittedBandPowerValues[37]), as.numeric(splittedBandPowerValues[38]), as.numeric(splittedBandPowerValues[39]), as.numeric(splittedBandPowerValues[40]), as.numeric(splittedBandPowerValues[41]),
               as.numeric(splittedBandPowerValues[42]), as.numeric(splittedBandPowerValues[43]), as.numeric(splittedBandPowerValues[44]), as.numeric(splittedBandPowerValues[45]), as.numeric(splittedBandPowerValues[46]),
               as.numeric(splittedBandPowerValues[47]), as.numeric(splittedBandPowerValues[48]), as.numeric(splittedBandPowerValues[49]), as.numeric(splittedBandPowerValues[50]), as.numeric(splittedBandPowerValues[51]),
               as.numeric(splittedBandPowerValues[52]), as.numeric(splittedBandPowerValues[53]), as.numeric(splittedBandPowerValues[54]), as.numeric(splittedBandPowerValues[55]), as.numeric(splittedBandPowerValues[56]),
               as.numeric(splittedBandPowerValues[57]), as.numeric(splittedBandPowerValues[58]), as.numeric(splittedBandPowerValues[59]), as.numeric(splittedBandPowerValues[60]), as.numeric(splittedBandPowerValues[61]),
               as.numeric(splittedBandPowerValues[62]), as.numeric(splittedBandPowerValues[63]), as.numeric(splittedBandPowerValues[64]), as.numeric(splittedBandPowerValues[65]), as.numeric(splittedBandPowerValues[66]),
               as.numeric(splittedBandPowerValues[67]), as.numeric(splittedBandPowerValues[68]), as.numeric(splittedBandPowerValues[69]), as.numeric(splittedBandPowerValues[70]), as.numeric(splittedBandPowerValues[71]));
      
      rawBandPowerDataFrameStage1Temp <- rawBandPowerDataFrameStage1                   
      rawBandPowerDataFrameStage1Temp[nrow(rawBandPowerDataFrameStage1) + 1, ] <- row
      rawBandPowerDataFrameStage1 <- rawBandPowerDataFrameStage1Temp
    }
    
  }
  
}

rawBandPowerDataFrameStage1Temp <- NULL
row <- NULL
tempDataFrame <- NULL
splittedBandPowerValues <- NULL
rowBandPowerValue <- NULL
rowTimeValue <- NULL

# ------------- Tage 2
rawBandPowerDataFrameStage2 <- NULL
rawBandPowerDataFrameStage2Temp <- NULL

for(i in 1:nrow(rawEmotivTrackingDataStage2)) {
  rowTimeValue <- rawEmotivTrackingDataStage2$time[i]
  rowBandPowerValue <- rawEmotivTrackingDataStage2$BandPowerDataMessage_CortexBrainComputerInterface[i]
  
  if (!is.null(rowBandPowerValue) && !is.na(rowBandPowerValue) && !is.nan(rowBandPowerValue) && length(rowBandPowerValue) && rowBandPowerValue != "")
  {
    splittedBandPowerValues <- stringr::str_split(rowBandPowerValue, ";")[[1]]
    splittedBandPowerValues <- stringr::str_replace(splittedBandPowerValues, ",", ".")
    
    if (is.null(rawBandPowerDataFrameStage2))
    {
      tempDataFrame <- data.frame("time" = c(as.numeric(rowTimeValue)), "AF3/theta" = c(as.numeric(splittedBandPowerValues[2])), "AF3/alpha" = c(as.numeric(splittedBandPowerValues[3])), "AF3/betaL" = c(as.numeric(splittedBandPowerValues[4])), "AF3/betaH" = c(as.numeric(splittedBandPowerValues[5])), "AF3/gamma" = c(as.numeric(splittedBandPowerValues[6])),
                                  "F7/theta" = c(as.numeric(splittedBandPowerValues[7])), "F7/alpha" = c(as.numeric(splittedBandPowerValues[8])),"F7/betaL" = c(as.numeric(splittedBandPowerValues[9])),"F7/betaH" = c(as.numeric(splittedBandPowerValues[10])),"F7/gamma" = c(as.numeric(splittedBandPowerValues[11])),
                                  "F3/theta" = c(as.numeric(splittedBandPowerValues[12])), "F3/alpha" = c(as.numeric(splittedBandPowerValues[13])),"F3/betaL" = c(as.numeric(splittedBandPowerValues[14])),"F3/betaH" = c(as.numeric(splittedBandPowerValues[15])),"F3/gamma" = c(as.numeric(splittedBandPowerValues[16])),
                                  "FC5/theta" = c(as.numeric(splittedBandPowerValues[17])),"FC5/alpha" = c(as.numeric(splittedBandPowerValues[18])),"FC5/betaL" = c(as.numeric(splittedBandPowerValues[19])),"FC5/betaH" = c(as.numeric(splittedBandPowerValues[20])),"FC5/gamma" = c(as.numeric(splittedBandPowerValues[21])),
                                  "T7/theta" = c(as.numeric(splittedBandPowerValues[22])), "T7/alpha" = c(as.numeric(splittedBandPowerValues[23])),"T7/betaL" = c(as.numeric(splittedBandPowerValues[24])),"T7/betaH" = c(as.numeric(splittedBandPowerValues[25])),"T7/gamma" = c(as.numeric(splittedBandPowerValues[26])),
                                  "P7/theta" = c(as.numeric(splittedBandPowerValues[27])), "P7/alpha" = c(as.numeric(splittedBandPowerValues[28])),"P7/betaL" = c(as.numeric(splittedBandPowerValues[29])),"P7/betaH" = c(as.numeric(splittedBandPowerValues[30])),"P7/gamma" = c(as.numeric(splittedBandPowerValues[31])),
                                  "O1/theta" = c(as.numeric(splittedBandPowerValues[32])), "O1/alpha" = c(as.numeric(splittedBandPowerValues[33])),"O1/betaL" = c(as.numeric(splittedBandPowerValues[34])),"O1/betaH" = c(as.numeric(splittedBandPowerValues[35])),"O1/gamma" = c(as.numeric(splittedBandPowerValues[36])),
                                  "O2/theta" = c(as.numeric(splittedBandPowerValues[37])), "O2/alpha" = c(as.numeric(splittedBandPowerValues[38])),"O2/betaL" = c(as.numeric(splittedBandPowerValues[39])),"O2/betaH" = c(as.numeric(splittedBandPowerValues[40])),"O2/gamma" = c(as.numeric(splittedBandPowerValues[41])),
                                  "P8/theta" = c(as.numeric(splittedBandPowerValues[42])), "P8/alpha" = c(as.numeric(splittedBandPowerValues[43])),"P8/betaL" = c(as.numeric(splittedBandPowerValues[44])),"P8/betaH" = c(as.numeric(splittedBandPowerValues[45])),"P8/gamma" = c(as.numeric(splittedBandPowerValues[46])),
                                  "T8/theta" = c(as.numeric(splittedBandPowerValues[47])), "T8/alpha" = c(as.numeric(splittedBandPowerValues[48])),"T8/betaL" = c(as.numeric(splittedBandPowerValues[49])),"T8/betaH" = c(as.numeric(splittedBandPowerValues[50])),"T8/gamma" = c(as.numeric(splittedBandPowerValues[51])),
                                  "FC6/theta" = c(as.numeric(splittedBandPowerValues[52])),"FC6/alpha" = c(as.numeric(splittedBandPowerValues[53])),"FC6/betaL" = c(as.numeric(splittedBandPowerValues[54])),"FC6/betaH" = c(as.numeric(splittedBandPowerValues[55])),"FC6/gamma" = c(as.numeric(splittedBandPowerValues[56])),
                                  "F4/theta" = c(as.numeric(splittedBandPowerValues[57])), "F4/alpha" = c(as.numeric(splittedBandPowerValues[58])),"F4/betaL" = c(as.numeric(splittedBandPowerValues[59])),"F4/betaH" = c(as.numeric(splittedBandPowerValues[60])),"F4/gamma" = c(as.numeric(splittedBandPowerValues[61])),
                                  "F8/theta" = c(as.numeric(splittedBandPowerValues[62])), "F8/alpha" = c(as.numeric(splittedBandPowerValues[63])),"F8/betaL" = c(as.numeric(splittedBandPowerValues[64])),"F8/betaH" = c(as.numeric(splittedBandPowerValues[65])),"F8/gamma" = c(as.numeric(splittedBandPowerValues[66])),
                                  "AF4/theta" = c(as.numeric(splittedBandPowerValues[67])),"AF4/alpha" = c(as.numeric(splittedBandPowerValues[68])),"AF4/betaL" = c(as.numeric(splittedBandPowerValues[69])),"AF4/betaH" = c(as.numeric(splittedBandPowerValues[70])),"AF4/gamma" = c(as.numeric(splittedBandPowerValues[71])));
      
      rawBandPowerDataFrameStage2 <- tempDataFrame
    }
    else
    {
      row <- c(as.numeric(rowTimeValue), as.numeric(splittedBandPowerValues[2]), as.numeric(splittedBandPowerValues[3]), as.numeric(splittedBandPowerValues[4]), as.numeric(splittedBandPowerValues[5]), as.numeric(splittedBandPowerValues[6]),
               as.numeric(splittedBandPowerValues[7]),  as.numeric(splittedBandPowerValues[8]),  as.numeric(splittedBandPowerValues[9]),  as.numeric(splittedBandPowerValues[10]), as.numeric(splittedBandPowerValues[11]),
               as.numeric(splittedBandPowerValues[12]), as.numeric(splittedBandPowerValues[13]), as.numeric(splittedBandPowerValues[14]), as.numeric(splittedBandPowerValues[15]), as.numeric(splittedBandPowerValues[16]),
               as.numeric(splittedBandPowerValues[17]), as.numeric(splittedBandPowerValues[18]), as.numeric(splittedBandPowerValues[19]), as.numeric(splittedBandPowerValues[20]), as.numeric(splittedBandPowerValues[21]),
               as.numeric(splittedBandPowerValues[22]), as.numeric(splittedBandPowerValues[23]), as.numeric(splittedBandPowerValues[24]), as.numeric(splittedBandPowerValues[25]), as.numeric(splittedBandPowerValues[26]),
               as.numeric(splittedBandPowerValues[27]), as.numeric(splittedBandPowerValues[28]), as.numeric(splittedBandPowerValues[29]), as.numeric(splittedBandPowerValues[30]), as.numeric(splittedBandPowerValues[31]),
               as.numeric(splittedBandPowerValues[32]), as.numeric(splittedBandPowerValues[33]), as.numeric(splittedBandPowerValues[34]), as.numeric(splittedBandPowerValues[35]), as.numeric(splittedBandPowerValues[36]),
               as.numeric(splittedBandPowerValues[37]), as.numeric(splittedBandPowerValues[38]), as.numeric(splittedBandPowerValues[39]), as.numeric(splittedBandPowerValues[40]), as.numeric(splittedBandPowerValues[41]),
               as.numeric(splittedBandPowerValues[42]), as.numeric(splittedBandPowerValues[43]), as.numeric(splittedBandPowerValues[44]), as.numeric(splittedBandPowerValues[45]), as.numeric(splittedBandPowerValues[46]),
               as.numeric(splittedBandPowerValues[47]), as.numeric(splittedBandPowerValues[48]), as.numeric(splittedBandPowerValues[49]), as.numeric(splittedBandPowerValues[50]), as.numeric(splittedBandPowerValues[51]),
               as.numeric(splittedBandPowerValues[52]), as.numeric(splittedBandPowerValues[53]), as.numeric(splittedBandPowerValues[54]), as.numeric(splittedBandPowerValues[55]), as.numeric(splittedBandPowerValues[56]),
               as.numeric(splittedBandPowerValues[57]), as.numeric(splittedBandPowerValues[58]), as.numeric(splittedBandPowerValues[59]), as.numeric(splittedBandPowerValues[60]), as.numeric(splittedBandPowerValues[61]),
               as.numeric(splittedBandPowerValues[62]), as.numeric(splittedBandPowerValues[63]), as.numeric(splittedBandPowerValues[64]), as.numeric(splittedBandPowerValues[65]), as.numeric(splittedBandPowerValues[66]),
               as.numeric(splittedBandPowerValues[67]), as.numeric(splittedBandPowerValues[68]), as.numeric(splittedBandPowerValues[69]), as.numeric(splittedBandPowerValues[70]), as.numeric(splittedBandPowerValues[71]));
      
      rawBandPowerDataFrameStage2Temp <- rawBandPowerDataFrameStage2                   
      rawBandPowerDataFrameStage2Temp[nrow(rawBandPowerDataFrameStage2) + 1, ] <- row
      rawBandPowerDataFrameStage2 <- rawBandPowerDataFrameStage2Temp
    }
  }
}

rawBandPowerDataFrameStage2Temp <- NULL
row <- NULL
tempDataFrame <- NULL
splittedBandPowerValues <- NULL
rowBandPowerValue <- NULL
rowTimeValue <- NULL
