### All Uncanny Valley Answers
### ----------------------------------------------------------------------------
# filter out unnecessary rows
countColumns <- ncol(uncannyValleyAnswersRawData)
uncannyValleyAnswersSubset <- uncannyValleyAnswersRawData[3:countColumns]
# filter transposed rows and columns
uncannyValleyAnswersSubsetTrasponiert = t(uncannyValleyAnswersSubset)
# make a copy of transposed data frame
uncannyValleyAnswersSubsetTrasponiertRenamed <- uncannyValleyAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(uncannyValleyAnswersSubsetTrasponiertRenamed)<-unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])
uncannyValleyAnswersFilteredData <- uncannyValleyAnswersSubsetTrasponiertRenamed[!row.names(uncannyValleyAnswersSubsetTrasponiertRenamed)=='QuestionID', ]
# create data frame 
# data_frame$column <- as.numeric(as.character(data_frame$column)) # needs to be testet
countRows <- nrow(uncannyValleyAnswersFilteredData)
uncannyValleyAnswersFilteredData <- data.frame(as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 1]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 2]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 3]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 4]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 5]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 6]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 7]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 8]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 9]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 10]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 11]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 12]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 13]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 14]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 15]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 16]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 17]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 18]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 19]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 20]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 21]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 22]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 23]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 24]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 25]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 26]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 27]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 28]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 29]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 30]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 31]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 32]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 33]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 34]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 35]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 36]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 37]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 38]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 39]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 40]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 41]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 42]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 43]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 44]),
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 45]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 46]), 
                                               as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 47]), as.numeric(uncannyValleyAnswersFilteredData[0:countRows, 48]))

# rename columns and rows
colnames(uncannyValleyAnswersFilteredData) <- unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])
rownames(uncannyValleyAnswersFilteredData) <- rownames(uncannyValleyAnswersSubsetTrasponiertRenamed)[1:countRows+1]

# descriptive statistic

# length(uncannyValleyAnswersFilteredData$q1_Lloid)
# summary(uncannyValleyAnswersFilteredData)

allUncannyValleyStatisticResults <- describe(uncannyValleyAnswersFilteredData)

path <- file.path(condition,  "RResults/Questionnaires", "/")

pathCSV <- file.path(path,  "AllUncannyValleyStatisticResults_DataFrame.csv", "")
write.csv2(allUncannyValleyStatisticResults, pathCSV, row.names = TRUE)

# hist(uncannyValleyAnswersFilteredData$q1_Lloid,prob=T,main="q1_Lloid")
# points(density(uncannyValleyAnswersFilteredData$q1_Lloid),type="l",col="blue")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### All Uncanny Valley selected Condition Answers
### ----------------------------------------------------------------------------
# filter out unnecessary rows
countColumns <- ncol(uncannyValleyConditionAnswersRawData)
uncannyValleyAnswersSubset <- uncannyValleyConditionAnswersRawData[3:countColumns]
# filter transposed rows and columns
uncannyValleyAnswersSubsetTrasponiert = t(uncannyValleyAnswersSubset)
# make a copy of transposed data frame
uncannyValleyAnswersSubsetTrasponiertRenamed <- uncannyValleyAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(uncannyValleyAnswersSubsetTrasponiertRenamed)<-unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])
uncannyValleyConditionAnswersFilteredData <- uncannyValleyAnswersSubsetTrasponiertRenamed[!row.names(uncannyValleyAnswersSubsetTrasponiertRenamed)=='QuestionID', ]
# create data frame 
countRows <- nrow(uncannyValleyConditionAnswersFilteredData)
uncannyValleyConditionAnswersFilteredData <- data.frame(as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 1]),  as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 2]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 3]),  as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 4]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 5]),  as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 6]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 7]),  as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 8]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 9]),  as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 10]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 11]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 12]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 13]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 14]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 15]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 16]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 17]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 18]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 19]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 20]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 21]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 22]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 23]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 24]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 25]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 26]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 27]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 28]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 29]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 30]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 31]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 32]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 33]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 34]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 35]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 36]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 37]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 38]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 39]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 40]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 41]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 42]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 43]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 44]),
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 45]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 46]), 
                                                        as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 47]), as.numeric(uncannyValleyConditionAnswersFilteredData[0:countRows, 48]))
uncannyValleyConditionAnswersFilteredData
# rename columns and rows
colnames(uncannyValleyConditionAnswersFilteredData) <- unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])
rownames(uncannyValleyConditionAnswersFilteredData) <- rownames(uncannyValleyAnswersSubsetTrasponiertRenamed)[1:countRows+1]

# descriptive statistic

# length(uncannyValleyAnswersFilteredData$q1_Lloid)
# summary(uncannyValleyAnswersFilteredData)

allUncannyValleyConditionStatisticResults <- describe(uncannyValleyConditionAnswersFilteredData)

path <- file.path(condition,  "RResults/Questionnaires", "/")

pathCSV <- file.path(path,  "AllUncannyValleyConditionStatisticResults_DataFrame.csv", "")
write.csv2(allUncannyValleyConditionStatisticResults, pathCSV, row.names = TRUE)

#hist(uncannyValleyConditionAnswersFilteredData$q1_Lloid,prob=T,main="q1_Lloid")
#points(density(uncannyValleyConditionAnswersFilteredData$q1_Lloid),type="l",col="blue")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### All SSQ Answers
### ----------------------------------------------------------------------------
# filter out unnecessary rows
countColumns <- ncol(SSQAnswersRawData)
SSQAnswersSubset <- SSQAnswersRawData[3:countColumns]
# filter transposed rows and columns
SSQAnswersSubsetTrasponiert = t(SSQAnswersSubset)
# make a copy of transposed data frame
SSQAnswersSubsetTrasponiertRenamed <- SSQAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(SSQAnswersSubsetTrasponiertRenamed)<-unlist(SSQAnswersSubsetTrasponiertRenamed[1, ])
SSQAnswersFilteredData <- SSQAnswersSubsetTrasponiertRenamed[!row.names(SSQAnswersSubsetTrasponiertRenamed)=='QuestionID', ]
# create data frame 
countRows <- nrow(SSQAnswersFilteredData)
SSQAnswersFilteredData <- data.frame(as.numeric(SSQAnswersFilteredData[0:countRows, 1]),  as.numeric(SSQAnswersFilteredData[0:countRows, 2]), 
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 3]),  as.numeric(SSQAnswersFilteredData[0:countRows, 4]), 
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 5]),  as.numeric(SSQAnswersFilteredData[0:countRows, 6]),
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 7]),  as.numeric(SSQAnswersFilteredData[0:countRows, 8]), 
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 9]),  as.numeric(SSQAnswersFilteredData[0:countRows, 10]), 
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 11]), as.numeric(SSQAnswersFilteredData[0:countRows, 12]),
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 13]), as.numeric(SSQAnswersFilteredData[0:countRows, 14]),
                                     as.numeric(SSQAnswersFilteredData[0:countRows, 15]), as.numeric(SSQAnswersFilteredData[0:countRows, 16]))
# rename columns and rows
colnames(SSQAnswersFilteredData) <- unlist(SSQAnswersSubsetTrasponiertRenamed[1, ])
rownames(SSQAnswersFilteredData) <- rownames(SSQAnswersSubsetTrasponiertRenamed)[1:countRows+1]
# descriptive statistic
allSSQStatisticResults <- describe(SSQAnswersFilteredData)

path <- file.path(condition,  "RResults/Questionnaires", "/")

pathCSV <- file.path(path,  "AllSSQStatisticResults_DataFrame.csv", "")
write.csv2(allSSQStatisticResults, pathCSV, row.names = TRUE)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### All SSQ Condition Answers
### ----------------------------------------------------------------------------
# filter out unnecessary rows
countColumns <- ncol(SSQConditionAnswersRawData)
SSQAnswersSubset <- SSQConditionAnswersRawData[3:countColumns]
# filter transposed rows and columns
SSQAnswersSubsetTrasponiert = t(SSQAnswersSubset)
# make a copy of transposed data frame
SSQAnswersSubsetTrasponiertRenamed <- SSQAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(SSQAnswersSubsetTrasponiertRenamed)<-unlist(SSQAnswersSubsetTrasponiertRenamed[1, ])
SSQConditionAnswersFilteredData <- SSQAnswersSubsetTrasponiertRenamed[!row.names(SSQAnswersSubsetTrasponiertRenamed)=='QuestionID', ]
# create data frame 
countRows <- nrow(SSQConditionAnswersFilteredData)
SSQConditionAnswersFilteredData <- data.frame(as.numeric(SSQConditionAnswersFilteredData[0:countRows, 1]),  as.numeric(SSQConditionAnswersFilteredData[0:countRows, 2]), 
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 3]),  as.numeric(SSQConditionAnswersFilteredData[0:countRows, 4]), 
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 5]),  as.numeric(SSQConditionAnswersFilteredData[0:countRows, 6]),
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 7]),  as.numeric(SSQConditionAnswersFilteredData[0:countRows, 8]), 
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 9]),  as.numeric(SSQConditionAnswersFilteredData[0:countRows, 10]), 
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 11]), as.numeric(SSQConditionAnswersFilteredData[0:countRows, 12]),
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 13]), as.numeric(SSQConditionAnswersFilteredData[0:countRows, 14]),
                                              as.numeric(SSQConditionAnswersFilteredData[0:countRows, 15]), as.numeric(SSQConditionAnswersFilteredData[0:countRows, 16]))
# rename columns and rows
colnames(SSQConditionAnswersFilteredData) <- unlist(SSQAnswersSubsetTrasponiertRenamed[1, ])
rownames(SSQConditionAnswersFilteredData) <- rownames(SSQAnswersSubsetTrasponiertRenamed)[1:countRows+1]
# descriptive statistic
allSSQConditionStatisticResults <- describe(SSQConditionAnswersFilteredData)

path <- file.path(condition,  "RResults/Questionnaires", "/")

pathCSV <- file.path(path,  "AllSSQConditionStatisticResults_DataFrame.csv", "")
write.csv2(allSSQConditionStatisticResults, pathCSV, row.names = TRUE)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------



### All MSSQ Answers
### ----------------------------------------------------------------------------
allMSSQStatisticResults <- describe(MSSQAnswersRawData)

path <- file.path(condition,  "RResults/Questionnaires", "/")

pathCSV <- file.path(path,  "AllMSSQStatisticResults_DataFrame.csv", "")
write.csv2(allMSSQStatisticResults, pathCSV, row.names = TRUE)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
