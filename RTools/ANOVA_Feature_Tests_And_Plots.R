# ANOVA Feature tests and plots
# ===================================

# install packages that are needed
#install.packages(c("ggplot2", "ggpubr", "tidyverse", "broom", "AICcmodavg"))

# using packages
library(ggplot2)
library(ggpubr)
library(tidyverse)
library(broom)
library(AICcmodavg)
library(Hmisc)
library(car)
library(ggstatsplot)



# Test anova method
# ===================================

# read csv file as data frame
#allFeaturesTrackedFromStage1 <- read.csv2(file = 'All_Participents_DataFrame.csv')

# create descriptive statistic
#summary(allFeaturesTrackedFromStage1)

# ANOVA test one-way (overall variance of the data, statistically significant)
#one.way <- aov(HeartRate ~ time, data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(one.way)

# ANOVA test many-way (overall variance of the data, statistically significant)
#many.way <- aov(HeartRate ~ amps + AF3.betaH + AF3.gamma + eng + int + foc + LeftPupilDiameter + RightPupilDiameter + FixationCounter + FixationDuration + DegTimeLowQuality + TIMERSICalc, data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(many.way)


# ANOVA test one-way (overall variance of the data, statistically significant)
#any.way <- aov(HeartRate ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(any.way)

# ANOVA test one-way (overall variance of the data, statistically significant)
#any.way <- aov(DegTimeLowQuality ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(any.way)

# ANOVA test one-way (overall variance of the data, statistically significant)
#any.way <- aov(FixationDuration ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(any.way)

# ===================================


#boxplot(TIMERSICalc ~ foc, data = allFeaturesTrackedFromStage1)

#### conscientious
# ===================================
# manuell clustert (0 = conscientious, 1 = none-conscientious)
# =======
# read csv file as data frame
allConscientiousFeaturesTrackedFromStage1 <- read.csv2(file = './Condition A/RResults/All_Participents_Stage1_DataFrame.csv')
allNoneConscientiousFeaturesTrackedFromStage1 <- read.csv2(file = './Condition B/RResults/All_Participents_Stage1_DataFrame.csv')
allNoneFeaturesTrackedFromStage1 <- read.csv2(file = './Condition C/RResults/All_Participents_Stage1_DataFrame.csv')
allFeaturesTrackedFromStage1 <- read.csv2(file = 'All_Participents_Stage1_DataFrame.csv')
allMeanFeaturesTrackedFromStage1 <- read.csv2(file = 'All_Participents_Mean_Stage1_DataFrame.csv')

# Condition A
temp <- allConscientiousFeaturesTrackedFromStage1
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
# eeg band power column number = 70 and begin with 10
allConscientiousFeaturesTrackedFromStage1 <- temp
columnConscientiousCounter <- ncol(allConscientiousFeaturesTrackedFromStage1)
barplot(colSums(allConscientiousFeaturesTrackedFromStage1[,(34):103]))
barplot(colSums(allConscientiousFeaturesTrackedFromStage1[,(columnConscientiousCounter - 4):columnConscientiousCounter]))
barplot(colSums(allConscientiousFeaturesTrackedFromStage1[,(columnConscientiousCounter - 2):columnConscientiousCounter]))

# Condition B
temp <- allNoneConscientiousFeaturesTrackedFromStage1
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
allNoneConscientiousFeaturesTrackedFromStage1 <- temp
columnNoneConscientiousCounter <- ncol(allNoneConscientiousFeaturesTrackedFromStage1)
barplot(colSums(allNoneConscientiousFeaturesTrackedFromStage1[,(34):103]))
barplot(colSums(allNoneConscientiousFeaturesTrackedFromStage1[,(columnConscientiousCounter - 4):columnConscientiousCounter]))
barplot(colSums(allNoneConscientiousFeaturesTrackedFromStage1[,(columnConscientiousCounter - 2):columnConscientiousCounter]))

# Condition C
temp <- allNoneFeaturesTrackedFromStage1
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
allNoneFeaturesTrackedFromStage1 <- temp
columnNoneCounter <- ncol(allNoneFeaturesTrackedFromStage1)
barplot(colSums(allNoneFeaturesTrackedFromStage1[,(34):103]))
barplot(colSums(allNoneFeaturesTrackedFromStage1[,(columnConscientiousCounter - 4):columnConscientiousCounter]))
barplot(colSums(allNoneFeaturesTrackedFromStage1[,(columnConscientiousCounter - 2):columnConscientiousCounter]))
write.csv2(allNoneFeaturesTrackedFromStage1, "All_Participents_Condition-C_WaveSum_DataFrame.csv", row.names = FALSE)


# Condition A-B-C
temp <- allFeaturesTrackedFromStage1
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
allFeaturesTrackedFromStage1 <- temp
columnAllCounter <- ncol(allFeaturesTrackedFromStage1)
barplot(colSums(allFeaturesTrackedFromStage1[,(34):103]))
barplot(colSums(allFeaturesTrackedFromStage1[,(columnAllCounter - 4):columnAllCounter]))
barplot(colSums(allFeaturesTrackedFromStage1[,(columnAllCounter - 2):columnAllCounter]))
write.csv2(allFeaturesTrackedFromStage1, "All_Participents_WaveSum_DataFrame.csv", row.names = FALSE)

# Condition A-B-C Mean
temp <- allMeanFeaturesTrackedFromStage1
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
allMeanFeaturesTrackedFromStage1 <- temp
columnAllMeanCounter <- ncol(allMeanFeaturesTrackedFromStage1)
barplot(colSums(allMeanFeaturesTrackedFromStage1[,(34):103]))
barplot(colSums(allMeanFeaturesTrackedFromStage1[,(columnAllMeanCounter - 4):columnAllMeanCounter]))
barplot(colSums(allMeanFeaturesTrackedFromStage1[,(columnAllMeanCounter - 2):columnAllMeanCounter]))
write.csv2(allMeanFeaturesTrackedFromStage1, "All_Participents_WaveSum_Mean_DataFrame.csv", row.names = FALSE)


rowConscientiousCounter     <- nrow(allConscientiousFeaturesTrackedFromStage1)
rowNoneConscientiousCounter <- nrow(allNoneConscientiousFeaturesTrackedFromStage1)
#rowNoneCounter              <- nrow(allNoneFeaturesTrackedFromStage1)
rowAllCounter               <- nrow(allFeaturesTrackedFromStage1)

# ANOVA test one-way (overall variance of the data, statistically significant)
any.way <- aov(EvaluatedGlobalTIMERSICalc ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
summary(any.way)


# ANOVA test one-way (overall variance of the data, statistically significant)
any.way <- aov(DegTimeLowQuality ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
summary(any.way)


#allFeaturesTrackedFromStage1$theta <- 0
#allFeaturesTrackedFromStage1$alpha <- 0
#allFeaturesTrackedFromStage1$betaL <- 0
#allFeaturesTrackedFromStage1$betaH <- 0
#allFeaturesTrackedFromStage1$gamma <- 0

#allFeaturesTrackedFromStage1$theta[1:rowCounter] <- allConscientiousFeaturesTrackedFromStage1$theta
#allFeaturesTrackedFromStage1$alpha[1:rowCounter] <- allConscientiousFeaturesTrackedFromStage1$alpha
#allFeaturesTrackedFromStage1$betaL[1:rowCounter] <- allConscientiousFeaturesTrackedFromStage1$betaL
#allFeaturesTrackedFromStage1$betaH[1:rowCounter] <- allConscientiousFeaturesTrackedFromStage1$betaH
#allFeaturesTrackedFromStage1$gamma[1:rowCounter] <- allConscientiousFeaturesTrackedFromStage1$gamma

#allFeaturesTrackedFromStage1$theta[rowCounter+1:rowNoneCounter] <- allNoneConscientiousFeaturesTrackedFromStage1$theta
#allFeaturesTrackedFromStage1$alpha[rowCounter+1:rowNoneCounter] <- allNoneConscientiousFeaturesTrackedFromStage1$alpha
#allFeaturesTrackedFromStage1$betaL[rowCounter+1:rowNoneCounter] <- allNoneConscientiousFeaturesTrackedFromStage1$betaL
#allFeaturesTrackedFromStage1$betaH[rowCounter+1:rowNoneCounter] <- allNoneConscientiousFeaturesTrackedFromStage1$betaH
#allFeaturesTrackedFromStage1$gamma[rowCounter+1:rowNoneCounter] <- allNoneConscientiousFeaturesTrackedFromStage1$gamma


# normalize data frame
# -------------------------------------------------------------------------------------------------------------------------------
#allFeaturesTrackedFromStage1 <- as.data.frame(scale(allFeaturesTrackedFromStage1)) 

# -- get all columns type infos too
# summary.default(allConscientiousFeaturesTrackedFromStage1)


# Condition A
scale_ConscientiousFeaturesTrackedFromStage1 <- allConscientiousFeaturesTrackedFromStage1

scale_ConscientiousFeaturesTrackedFromStage1 <- subset(scale_ConscientiousFeaturesTrackedFromStage1, , -pId )
scale_ConscientiousFeaturesTrackedFromStage1 <- subset(scale_ConscientiousFeaturesTrackedFromStage1, , -STARTED )
scale_ConscientiousFeaturesTrackedFromStage1 <- subset(scale_ConscientiousFeaturesTrackedFromStage1, , -LASTDATA )
#summary.default(scale_ConscientiousFeaturesTrackedFromStage1)

scale_ConscientiousFeaturesTrackedFromStage1 <- as.data.frame(scale(scale_ConscientiousFeaturesTrackedFromStage1))
scale_columnConscientiousCounter <- ncol(scale_ConscientiousFeaturesTrackedFromStage1)
barplot(colSums(scale_ConscientiousFeaturesTrackedFromStage1[,(scale_columnConscientiousCounter - 4):scale_columnConscientiousCounter]))
barplot(colSums(scale_ConscientiousFeaturesTrackedFromStage1[,(scale_columnConscientiousCounter - 2):scale_columnConscientiousCounter]))

# Condition B
scale_NoneConscientiousFeaturesTrackedFromStage1 <- allNoneConscientiousFeaturesTrackedFromStage1

scale_NoneConscientiousFeaturesTrackedFromStage1 <- subset(scale_NoneConscientiousFeaturesTrackedFromStage1, , -pId  )
scale_NoneConscientiousFeaturesTrackedFromStage1 <- subset(scale_NoneConscientiousFeaturesTrackedFromStage1, , -STARTED )
scale_NoneConscientiousFeaturesTrackedFromStage1 <- subset(scale_NoneConscientiousFeaturesTrackedFromStage1, , -LASTDATA )

scale_NoneConscientiousFeaturesTrackedFromStage1 <- as.data.frame(scale(scale_NoneConscientiousFeaturesTrackedFromStage1))
scale_columnNoneConscientiousCounter <- ncol(scale_NoneConscientiousFeaturesTrackedFromStage1)
barplot(colSums(scale_NoneConscientiousFeaturesTrackedFromStage1[,(scale_columnNoneConscientiousCounter - 4):scale_columnNoneConscientiousCounter]))
barplot(colSums(scale_NoneConscientiousFeaturesTrackedFromStage1[,(scale_columnNoneConscientiousCounter - 2):scale_columnNoneConscientiousCounter]))

# Condition C
scale_NoneFeaturesTrackedFromStage1 <- allNoneFeaturesTrackedFromStage1

scale_NoneFeaturesTrackedFromStage1 <- subset(scale_NoneFeaturesTrackedFromStage1, , -pId )
scale_NoneFeaturesTrackedFromStage1 <- subset(scale_NoneFeaturesTrackedFromStage1, , -STARTED )
scale_NoneFeaturesTrackedFromStage1 <- subset(scale_NoneFeaturesTrackedFromStage1, , -LASTDATA )

scale_NoneFeaturesTrackedFromStage1 <- as.data.frame(scale(scale_NoneFeaturesTrackedFromStage1))
scale_columnNoneCounter <- ncol(scale_NoneFeaturesTrackedFromStage1)
barplot(colSums(scale_NoneFeaturesTrackedFromStage1[,(scale_columnNoneCounter - 4):scale_columnNoneCounter]))
barplot(colSums(scale_NoneFeaturesTrackedFromStage1[,(scale_columnNoneCounter - 2):scale_columnNoneCounter]))

con_columns_sum <- colSums(scale_ConscientiousFeaturesTrackedFromStage1[,(scale_columnConscientiousCounter - 2):scale_columnConscientiousCounter])
none_con_columns_sum <- colSums(scale_NoneConscientiousFeaturesTrackedFromStage1[,(scale_columnNoneConscientiousCounter - 2):scale_columnNoneConscientiousCounter])
none_features_columns_sum <- colSums(scale_NoneFeaturesTrackedFromStage1[,(scale_columnNoneCounter - 2):scale_columnNoneCounter])

print(con_columns_sum)
print(none_con_columns_sum)
print(none_features_columns_sum)

col_names <- colnames(scale_ConscientiousFeaturesTrackedFromStage1[,(scale_columnConscientiousCounter - 2):scale_columnConscientiousCounter])
print(col_names)
data <- data.frame(values = c(8.383442e-14, -4.973799e-14, 1.473821e-14, 
                              -8.710047e-14, 3.001072e-14, -2.008810e-14, 
                              7.211245e-14, -5.431506e-14, 9.946688e-14),
                   group = rep(c("Conscientious",
                                 "None-Conscientious",
                                 "Free"), each = 3), subgroup = col_names)

print(data)

data_base <- reshape(data, idvar = "subgroup", timevar = "group", direction = "wide")
print(data_base)

row.names(data_base) <- data_base$subgroup
data_base <- data_base[ , 2:ncol(data_base)]
colnames(data_base) <- c("Conscientious", "None-Conscientious", "Free")
data_base <- as.matrix(data_base)
print(data_base)                                        

# ----- group data barplot
barplot(height = data_base, beside = TRUE)
ggplot(data, aes(x = group, y = values, fill = subgroup)) + geom_bar(stat = "identity", position = "dodge")


# only if Condition A-B manuel set to clusters (Conscientious = 0 and None-Conscientious = 1) 
# =======================================================================================================================
allFeaturesTrackedFromStage1$Conscientious <- 1
allFeaturesTrackedFromStage1$Conscientious[1:rowConscientiousCounter] <- 0
write.csv2(allFeaturesTrackedFromStage1, "All_Participents_Clusterd_WaveSum_DataFrame.csv", row.names = FALSE)


# with out pIDs (14, 15, 16)
#allFeaturesTrackedFromStage1 <- allFeaturesTrackedFromStage1[ !(allFeaturesTrackedFromStage1$pId %in% c(14,15,16)), ]

# create descriptive statistic
#summary(allFeaturesTrackedFromStage1)

# ANOVA test two-way (overall variance of the data, statistically significant)
#conscientious.way <- aov(Conscientious ~ ., data = allFeaturesTrackedFromStage1)

# create descriptive statistic
#summary(conscientious.way)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = RightMeanPupilDiameter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(RightMeanPupilDiameter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = LeftMeanPupilDiameter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(LeftMeanPupilDiameter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = CognitiveActivityRightPupilDiamter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(CognitiveActivityRightPupilDiamter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = CognitiveActivityLeftPupilDiamter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(CognitiveActivityLeftPupilDiamter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = MEDIANForTRSI, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(MEDIANForTRSI ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = TIMERSICalc, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(TIMERSICalc ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = AbsoluteDerivationOfResponseValue, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(AbsoluteDerivationOfResponseValue ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = StandardDeviationStraightLineAnswer, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(StandardDeviationStraightLineAnswer ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = DegTimeLowQuality, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(DegTimeLowQuality ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = SaccadeCounter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(SaccadeCounter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = TotalFixationDuration, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(TotalFixationDuration ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = TotalFixationCounter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(TotalFixationCounter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = RightPupilDiameter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(RightPupilDiameter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = LeftPupilDiameter, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(LeftPupilDiameter ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = eng, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(eng ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = exc, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(alpha ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = str, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(str ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = rel, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(rel ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = int, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(int ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = foc, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(foc ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = HeartRate, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(HeartRate ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = amps, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(amps ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = theta, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(theta ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = alpha, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(alpha ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = betaL, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(betaL ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = betaH, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(betaH ~ Conscientious, data = allFeaturesTrackedFromStage1)

#ggplot(allFeaturesTrackedFromStage1) + aes(x = Conscientious, y = gamma, color = Conscientious) + geom_jitter() + theme(legend.position = "none")
# Boxplot
#boxplot(gamma ~ Conscientious, data = allFeaturesTrackedFromStage1)

# histogram
#hist(conscientious.way$residuals)
#hist.data.frame(allFeaturesTrackedFromStage1)
# QQ-plot
#library(car)
#qqPlot(conscientious.way$residuals, id = FALSE) # id = FALSE to remove point identification


eegHist<- allFeaturesTrackedFromStage1[1:rowAllCounter, c('theta', 'alpha', 'betaL', 'betaH', 'gamma')]
par(las = 1) # all axis labels horizontal
boxplot(eegHist, main = "boxplot(*, horizontal = FALSE)", horizontal = FALSE)
#
#hist.data.frame(eegHist)
#
#ggplot(gather(eegHist, cols, value), aes(x = value)) + geom_histogram(binwidth = 20, bins=5 ) + facet_grid(.~cols)


#eegHist<- allFeaturesTrackedFromStage1[rowAllCounter + 1:rowNoneCounter, c('theta', 'alpha', 'betaL', 'betaH', 'gamma')]
#
#par(las = 1) # all axis labels horizontal
#boxplot(eegHist, main = "boxplot(*, horizontal = FALSE)", horizontal = FALSE)
#
##ggplot(gather(eegHist, cols, value), aes(x = value)) + geom_histogram(binwidth = 20, bins=5 ) + facet_grid(.~cols)


#eegHist<- allFeaturesTrackedFromStage1[, c('theta', 'alpha', 'betaL', 'betaH', 'gamma', 'Conscientious')]
#
##ggplot(eegHist, aes(x = Conscientious)) + geom_bar()
#par(las = 1) # all axis labels horizontal
#boxplot(eegHist, main = "boxplot(*, horizontal = FALSE)", horizontal = FALSE)
#
#hist.data.frame(eegHist)
#
##ggplot(gather(eegHist, cols, value), aes(x = value)) + geom_histogram(binwidth = 20, bins=5 ) + facet_grid(.~cols)



### Validity Score mit DegTimeLowQuality
# gamma
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = gamma, type = "parametric", # ANOVA or Kruskal-Wallis
  var.equal = TRUE, # ANOVA or Welch ANOVA
  plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# betaH
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = betaH, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# HeartRate
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = HeartRate, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# SD1SD2Ratio
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = SD1SD2Ratio, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# LFHFRatio
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = LFHFRatio, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# CognitiveActivityRightPupilDiamter
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = CognitiveActivityRightPupilDiamter, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# TotalFixationDuration
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = TotalFixationDuration, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# SaccadeCounter
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = DegTimeLowQuality, y = SaccadeCounter, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

### Validity Score mit EvaluatedTIMERSICalc
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = gamma, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# betaH
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = betaH, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# HeartRate
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = HeartRate, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# SD1SD2Ratio
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = SD1SD2Ratio, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# LFHFRatio
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = LFHFRatio, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# CognitiveActivityRightPupilDiamter
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = CognitiveActivityRightPupilDiamter, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# TotalFixationDuration
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = TotalFixationDuration, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)
# SaccadeCounter
ggbetweenstats(data = allFeaturesTrackedFromStage1, x = EvaluatedGlobalTIMERSICalc, y = SaccadeCounter, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)



#  Shapiro-Wilk normality test
#shapiro.test(conscientious.way$residuals)

# ===================================