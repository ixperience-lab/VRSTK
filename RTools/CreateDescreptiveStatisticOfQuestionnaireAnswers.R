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

describe(uncannyValleyAnswersFilteredData)

# hist(uncannyValleyAnswersFilteredData$q1_Lloid,prob=T,main="q1_Lloid")
# points(density(uncannyValleyAnswersFilteredData$q1_Lloid),type="l",col="blue")


### All SSQ Answers
### ----------------------------------------------------------------------------
# filter out unnecessary rows
countColumns <- ncol(allSSQAnswersRawData)
allSSQAnswersSubset <- allSSQAnswersRawData[3:countColumns]
# filter transposed rows and columns
allSSQAnswersSubsetTrasponiert = t(allSSQAnswersSubset)
# make a copy of transposed data frame
allSSQAnswersSubsetTrasponiertRenamed <- allSSQAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(allSSQAnswersSubsetTrasponiertRenamed)<-unlist(allSSQAnswersSubsetTrasponiertRenamed[1, ])
allSSQAnswersFilteredData <- allSSQAnswersSubsetTrasponiertRenamed[!row.names(allSSQAnswersSubsetTrasponiertRenamed)=='QuestionID', ]
# create data frame 
countRows <- nrow(allSSQAnswersFilteredData)
allSSQAnswersFilteredData <- data.frame(as.numeric(allSSQAnswersFilteredData[0:countRows, 1]), as.numeric(allSSQAnswersFilteredData[0:countRows, 2]), 
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 3]), as.numeric(allSSQAnswersFilteredData[0:countRows, 4]), 
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 5]), as.numeric(allSSQAnswersFilteredData[0:countRows, 6]),
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 7]), as.numeric(allSSQAnswersFilteredData[0:countRows, 8]), 
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 9]), as.numeric(allSSQAnswersFilteredData[0:countRows, 10]), 
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 11]), as.numeric(allSSQAnswersFilteredData[0:countRows, 12]),
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 13]), as.numeric(allSSQAnswersFilteredData[0:countRows, 14]),
                                        as.numeric(allSSQAnswersFilteredData[0:countRows, 15]), as.numeric(allSSQAnswersFilteredData[0:countRows, 16]))

# rename columns and rows
colnames(allSSQAnswersFilteredData) <- unlist(allSSQAnswersSubsetTrasponiertRenamed[1, ])
rownames(allSSQAnswersFilteredData) <- rownames(allSSQAnswersSubsetTrasponiertRenamed)[1:countRows+1]
# descriptive statistic
describe(allSSQAnswersFilteredData)


### All MSSQ Answers
### ----------------------------------------------------------------------------
describe(allMSSQAnswersRawData)
