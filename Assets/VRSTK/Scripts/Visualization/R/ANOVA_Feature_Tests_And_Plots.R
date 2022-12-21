# ANOVA Feature tests and plots
# ===================================
# script for creating full features data set that is made by buildautomationprocess script

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
library(psych)

# custom function to implement min max scaling
minMax <- function(x) {
  if((max(x) - min(x)) == 0) {
    x <- 0
  }else{
    (x - min(x)) / (max(x) - min(x))
  }
}

# ANOVA Method
# ===================================

# read data (baseline and task)
input_stage_0 <- read.csv2(file = './All_Participents_Stage0_DataFrame.csv')
input_stage_1 <- read.csv2(file = './All_Participents_Stage1_DataFrame.csv')

input_stage_1[input_stage_1$EvaluatedGlobalTIMERSICalc > 0, ]$EvaluatedGlobalTIMERSICalc <- 1

# anova eeg
# ---------
# sum EEG bandpower waves from each sensor on headset
temp <- input_stage_0
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
input_stage_0 <- temp

#print(head(input_stage_0$theta, 5))
input_stage_0$theta_scaled <- scale(input_stage_0$theta)
input_stage_0$alpha_scaled <- scale(input_stage_0$alpha)
input_stage_0$betaL_scaled <- scale(input_stage_0$betaL)
input_stage_0$betaH_scaled <- scale(input_stage_0$betaH)
input_stage_0$gamma_scaled <- scale(input_stage_0$gamma)
#print(head(input_stage_0$theta_scaled, 5))

temp <- input_stage_1
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
input_stage_1 <- temp

input_stage_1$theta_scaled <- scale(input_stage_1$theta)
input_stage_1$alpha_scaled <- scale(input_stage_1$alpha)
input_stage_1$betaL_scaled <- scale(input_stage_1$betaL)
input_stage_1$betaH_scaled <- scale(input_stage_1$betaH)
input_stage_1$gamma_scaled <- scale(input_stage_1$gamma)

# ANOVA test one-way (overall variance of the data, statistically significant)
descriptive_statistic_result <- describeBy(theta_scaled+alpha_scaled+betaL_scaled+betaH_scaled+gamma_scaled ~ 
                                           EvaluatedGlobalTIMERSICalc+DegTimeLowQuality+EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, 
                                           data = input_stage_1)
capture.output(descriptive_statistic_result, file = "./descriptive_statistic_eeg_result.txt")

eeg.one.way <- aov(theta_scaled+alpha_scaled+betaL_scaled+betaH_scaled+gamma_scaled ~ 
                   EvaluatedGlobalTIMERSICalc+DegTimeLowQuality+EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, 
                   data = input_stage_1)
# create descriptive statistic
anova_summary <- summary(eeg.one.way)
capture.output(anova_summary, file = "./anova_eeg_summary_results.txt")
# normalverteilung der residuan
plot(eeg.one.way, 2)

# anova hrv
# ---------
# ANOVA test one-way (overall variance of the data, statistically significant)
descriptive_statistic_result <- describeBy(LFHFRatio+SD1SD2Ratio ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality+EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, data = input_stage_1)
capture.output(descriptive_statistic_result, file = "./descriptive_statistic_hrv_result.txt")
hrv.one.way <- aov(LFHFRatio+SD1SD2Ratio ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality+EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, data = input_stage_1)
# create descriptive statistic
anova_summary <- summary(hrv.one.way)
capture.output(anova_summary, file = "./anova_hrv_summary_results.txt")
# normalverteilung der residuan
plot(hrv.one.way, 2)

# anova eda
# ---------
input_stage_1$FilteredValueInMicroSiemens_scaled <- scale(input_stage_1$FilteredValueInMicroSiemens)
# ANOVA test one-way (overall variance of the data, statistically significant)
descriptive_statistic_result <- describeBy(FilteredValueInMicroSiemens_scaled ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality, data = input_stage_1)
capture.output(descriptive_statistic_result, file = "./descriptive_statistic_eda_result.txt")
eda.one.way <- aov(FilteredValueInMicroSiemens_scaled ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality, data = input_stage_1)
# create descriptive statistic
anova_summary <- summary(hrv.one.way)
capture.output(anova_summary, file = "./anova_eda_summary_results.txt")
# normalverteilung der residuan
plot(eda.one.way, 2)


# anova eye
# ---------
input_stage_1$LeftPercentChangePupilDialtion <- 0.0
input_stage_1$RightPercentChangePupilDialtion <- 0.0

seq <- c( 1, 2, 3, 4, 5, 6, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 31, 34, 21, 22, 23, 24, 25, 26, 27, 28, 29 )
for (i in seq){
  pId <- i
  # stage 0
  base_line_mean_l_pupil_diameter <- mean(input_stage_0[input_stage_0$pId == pId, ]$LeftPupilDiameter)
  base_line_mean_r_pupil_diameter <- mean(input_stage_0[input_stage_0$pId == pId, ]$RightPupilDiameter)
  # stage 1
  input_stage_1[input_stage_1$pId == pId, ]$LeftPercentChangePupilDialtion  <- (input_stage_1[input_stage_1$pId == pId, ]$LeftPupilDiameter - base_line_mean_l_pupil_diameter) / base_line_mean_l_pupil_diameter
  input_stage_1[input_stage_1$pId == pId, ]$RightPercentChangePupilDialtion  <- (input_stage_1[input_stage_1$pId == pId, ]$RightPupilDiameter - base_line_mean_r_pupil_diameter) / base_line_mean_r_pupil_diameter
}

input_stage_1$LeftPercentChangePupilDialtion_scaled <- scale(input_stage_1$LeftPercentChangePupilDialtion)
input_stage_1$RightPercentChangePupilDialtion_scaled <- scale(input_stage_1$RightPercentChangePupilDialtion)

input_stage_1$TotalFixationCounter_scaled <- scale(input_stage_1$TotalFixationCounter)
input_stage_1$SaccadeCounter_scaled <- scale(input_stage_1$SaccadeCounter)

descriptive_statistic_result <- describeBy(LeftPercentChangePupilDialtion_scaled+RightPercentChangePupilDialtion_scaled+TotalFixationCounter_scaled+SaccadeCounter_scaled ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality, data = input_stage_1)
boxplot(LeftPercentChangePupilDialtion_scaled+RightPercentChangePupilDialtion_scaled+TotalFixationCounter_scaled+SaccadeCounter_scaled ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality, data = input_stage_1)
capture.output(descriptive_statistic_result, file = "./descriptive_statistic_eye_result.txt")
# ANOVA test one-way (overall variance of the data, statistically significant)
#eye.one.way <- aov(lm(EvaluatedGlobalTIMERSICalc+DegTimeLowQuality ~ LeftPercentChangePupilDialtion_scaled+RightPercentChangePupilDialtion_scaled+TotalFixationCounter_scaled+SaccadeCounter_scaled, data = input_stage_1))
eye.one.way <- aov(LeftPercentChangePupilDialtion_scaled+RightPercentChangePupilDialtion_scaled+TotalFixationCounter_scaled+SaccadeCounter_scaled ~ EvaluatedGlobalTIMERSICalc+DegTimeLowQuality+EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, data = input_stage_1)
# create descriptive statistic
anova_summary <- summary(eye.one.way)
capture.output(anova_summary, file = "./anova_eye_summary_results.txt")
# normalverteilung der residuan
plot(eye.one.way, 2)

#ggplot(input_stage_1) + aes(x = EvaluatedGlobalTIMERSICalc*DegTimeLowQuality, y = LeftPercentChangePupilDialtion_scaled+RightPercentChangePupilDialtion_scaled+TotalFixationCounter_scaled+SaccadeCounter_scaled, color = EvaluatedGlobalTIMERSICalc*DegTimeLowQuality) + geom_jitter() + theme(legend.position = "none")

ggbetweenstats(data = input_stage_1, x = EvaluatedGlobalTIMERSICalc, y = LeftPercentChangePupilDialtion_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

ggbetweenstats(data = input_stage_1, x = DegTimeLowQuality, y = LeftPercentChangePupilDialtion_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

ggbetweenstats(data = input_stage_1, x = EvaluatedGlobalTIMERSICalc, y = RightPercentChangePupilDialtion_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

ggbetweenstats(data = input_stage_1, x = DegTimeLowQuality, y = RightPercentChangePupilDialtion_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)


ggbetweenstats(data = input_stage_1, x = EvaluatedGlobalTIMERSICalc, y = TotalFixationCounter_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

ggbetweenstats(data = input_stage_1, x = DegTimeLowQuality, y = TotalFixationCounter_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)


ggbetweenstats(data = input_stage_1, x = EvaluatedGlobalTIMERSICalc, y = SaccadeCounter_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)

ggbetweenstats(data = input_stage_1, x = DegTimeLowQuality, y = SaccadeCounter_scaled, type = "parametric", # ANOVA or Kruskal-Wallis
               var.equal = TRUE, # ANOVA or Welch ANOVA
               plot.type = "box", pairwise.comparisons = TRUE, pairwise.display = "significant", centrality.plotting = FALSE, bf.message = FALSE)



#kruskal.test(input_stage_1$LeftPercentChangePupilDialtion_scaled, input_stage_1$EvaluatedGlobalTIMERSICalc)
#pairwise.wilcox.test(input_stage_1$LeftPercentChangePupilDialtion_scaled, input_stage_1$EvaluatedGlobalTIMERSICalc, paired = FALSE, p.adjust.method = "bonferroni")
#describeBy(input_stage_1$LeftPercentChangePupilDialtion_scaled, input_stage_1$EvaluatedGlobalTIMERSICalc)
#boxplot(LeftPercentChangePupilDialtion_scaled ~ EvaluatedGlobalTIMERSICalc, data = input_stage_1)


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
# Filter C Condition data frame
# alls cells with "NA" = 0, all rows with "NA" to cut out
allNoneFeaturesTrackedFromStage1 <- na.omit(allNoneFeaturesTrackedFromStage1)
# clean ids -> only numbers
allNoneFeaturesTrackedFromStage1$pId <- str_replace(allNoneFeaturesTrackedFromStage1$pId, "id-", "")
allNoneFeaturesTrackedFromStage1$pId <- str_replace(allNoneFeaturesTrackedFromStage1$pId, "b", "")
# switch time and id
numberOfColumns <- ncol(allNoneFeaturesTrackedFromStage1)
allNoneFeaturesTrackedFromStage1 <- allNoneFeaturesTrackedFromStage1[, c(2,1,3:numberOfColumns)] # leave the row index blank to keep all rows
# filter: remove unused/bad/const_to_zero/strings  columns
allNoneFeaturesTrackedFromStage1 <- subset(allNoneFeaturesTrackedFromStage1, select=-c(lex))
allNoneFeaturesTrackedFromStage1 <- subset(allNoneFeaturesTrackedFromStage1, select=-c(STARTED, LASTDATA, MAXPAGE, DegTimeThreshold, DegTimeThresholdForOnePage, DegTimeValueForOnePage)) # MAXPAGE is optional
# convert all columns to numeric
numberOfColumns <- ncol(allNoneFeaturesTrackedFromStage1)
allNoneFeaturesTrackedFromStage1[,1:numberOfColumns] <- lapply(allNoneFeaturesTrackedFromStage1[,1:numberOfColumns], function (x) as.numeric(x))
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

# only if Condition A-B manuel set to clusters (Conscientious = 0 and None-Conscientious = 1) 
# =======================================================================================================================
allFeaturesTrackedFromStage1$Conscientious <- 1
allFeaturesTrackedFromStage1$Conscientious[1:rowConscientiousCounter] <- 0

allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 21] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 22] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 23] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 26] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 27] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 28] <- 0
allFeaturesTrackedFromStage1$Conscientious[allFeaturesTrackedFromStage1$pId == 29] <- 0
write.csv2(allFeaturesTrackedFromStage1, "All_Participents_Clusterd_WaveSum_DataFrame.csv", row.names = FALSE)


#normalise data using custom function
#scale_ConscientiousFeatures <- allConscientiousFeaturesTrackedFromStage1
#scale_ConscientiousFeatures <- subset(scale_ConscientiousFeatures, , -pId )
#scale_ConscientiousFeatures <- subset(scale_ConscientiousFeatures, , -STARTED )
#scale_ConscientiousFeatures <- subset(scale_ConscientiousFeatures, , -LASTDATA )
#min_max_normalised_data_frame <- as.data.frame(lapply(scale_ConscientiousFeatures, minMax))
#selected_features_as_data_frame <- min_max_normalised_data_frame[1:nrow(min_max_normalised_data_frame), c('theta', 'alpha', 'betaL', 'betaH', 'gamma')]
#boxplot(selected_features_as_data_frame, main = "BCI sum vlaues", horizontal = FALSE)

#  Shapiro-Wilk normality test
#shapiro.test(conscientious.way$residuals)

# ===================================