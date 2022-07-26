#str(uncannyValleyAnswersRawData)
#summary(objects[0])

# filter out unnecessary rows -> [3:9] row actions
uncannyValleyAnswersSubset <- uncannyValleyAnswersRawData[3:9]
# filter transposed rows and columns
uncannyValleyAnswersSubsetTrasponiert = t(uncannyValleyAnswersSubset)
# make a copy of transposed data frame
uncannyValleyAnswersSubsetTrasponiertRenamed <- uncannyValleyAnswersSubsetTrasponiert
# rename columns to the first rows values, delete the first row and make a copy
colnames(uncannyValleyAnswersSubsetTrasponiertRenamed)<-unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])
uncannyValleyAnswersFilteredData <- uncannyValleyAnswersSubsetTrasponiertRenamed[!row.names(uncannyValleyAnswersSubsetTrasponiertRenamed)=='QuestionID', ]

# create data frame 
# data_frame$column <- as.numeric(as.character(data_frame$column)) # needs to be testet
uncannyValleyAnswersFilteredData <- data.frame(  as.numeric(uncannyValleyAnswersFilteredData[0:6, 1]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 2]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 3])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 4]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 5]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 6])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 7]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 8]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 9])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 10]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 11]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 12])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 13]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 14]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 15])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 16]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 17]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 18])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 19]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 20]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 21])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 22]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 23]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 24])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 25]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 26]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 27])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 28]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 29]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 30])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 31]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 32]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 33])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 34]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 35]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 36])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 37]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 38]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 39])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 40]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 41]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 42])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 43]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 44]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 45])
                                               , as.numeric(uncannyValleyAnswersFilteredData[0:6, 46]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 47]), as.numeric(uncannyValleyAnswersFilteredData[0:6, 48]))

colnames(uncannyValleyAnswersFilteredData) <- unlist(uncannyValleyAnswersSubsetTrasponiertRenamed[1, ])

# descriptive statistic

# length(uncannyValleyAnswersFilteredData$q1_Lloid)

# summary(uncannyValleyAnswersFilteredData)

describe(uncannyValleyAnswersFilteredData)

# hist(uncannyValleyAnswersFilteredData$q1_Lloid,prob=T,main="q1_Lloid")
# points(density(uncannyValleyAnswersFilteredData$q1_Lloid),type="l",col="blue")

