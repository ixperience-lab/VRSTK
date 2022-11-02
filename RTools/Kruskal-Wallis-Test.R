# kruskal.test Feature tests and plots
# ===================================

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

# ===================================
# read data (baseline and task)
input_stage_0 <- read.csv2(file = './All_Participents_Stage0_DataFrame.csv')
input_stage_1 <- read.csv2(file = './All_Participents_Stage1_DataFrame.csv')

input_stage_1[input_stage_1$EvaluatedGlobalTIMERSICalc > 0, ]$EvaluatedGlobalTIMERSICalc <- 1
input_stage_1$ValidityScore <- 0
input_stage_1[(input_stage_1$EvaluatedGlobalTIMERSICalc > 0) & (input_stage_1$DegTimeLowQuality > 0), ]$ValidityScore <- 1

# kruskal eeg
# ---------
# sum EEG bandpower waves from each sensor on headset
# Condition A
temp <- input_stage_0
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
input_stage_0 <- temp

#print(head(input_stage_0$theta, 5))
#input_stage_0$theta_scaled <- scale(input_stage_0$theta)
#input_stage_0$alpha_scaled <- scale(input_stage_0$alpha)
#input_stage_0$betaL_scaled <- scale(input_stage_0$betaL)
#input_stage_0$betaH_scaled <- scale(input_stage_0$betaH)
#input_stage_0$gamma_scaled <- scale(input_stage_0$gamma)
#print(head(input_stage_0$theta_scaled, 5))

temp <- input_stage_1
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
input_stage_1 <- temp

#input_stage_1$theta_scaled <- scale(input_stage_1$theta)
#input_stage_1$alpha_scaled <- scale(input_stage_1$alpha)
#input_stage_1$betaL_scaled <- scale(input_stage_1$betaL)
#input_stage_1$betaH_scaled <- scale(input_stage_1$betaH)
#input_stage_1$gamma_scaled <- scale(input_stage_1$gamma)

describeBy(theta ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$theta ~ input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$theta, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(theta ~ ValidityScore, data = input_stage_1)

describeBy(alpha ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$alpha, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$alpha, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(alpha ~ ValidityScore, data = input_stage_1)

describeBy(betaL ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$betaL ~ input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$betaL, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(betaL ~ ValidityScore, data = input_stage_1)

describeBy(betaH ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$betaH, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$betaH, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(betaH ~ ValidityScore, data = input_stage_1)

describeBy(gamma ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$gamma, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$gamma, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(gamma ~ ValidityScore, data = input_stage_1)


# kruskal hrv 
# ------
describeBy(LFHFRatio ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$LFHFRatio, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$LFHFRatio, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(LFHFRatio ~ ValidityScore, data = input_stage_1)

describeBy(SD1SD2Ratio ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$SD1SD2Ratio, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$SD1SD2Ratio, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(SD1SD2Ratio ~ ValidityScore, data = input_stage_1)

# kruskal eye
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

#input_stage_1$LeftPercentChangePupilDialtion_scaled <- scale(input_stage_1$LeftPercentChangePupilDialtion)
#input_stage_1$RightPercentChangePupilDialtion_scaled <- scale(input_stage_1$RightPercentChangePupilDialtion)

#input_stage_1$TotalFixationCounter_scaled <- scale(input_stage_1$TotalFixationCounter)
#input_stage_1$SaccadeCounter_scaled <- scale(input_stage_1$SaccadeCounter)

describeBy(LeftPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$LeftPercentChangePupilDialtion, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$LeftPercentChangePupilDialtion, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(LeftPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)

describeBy(RightPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$RightPercentChangePupilDialtion, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$RightPercentChangePupilDialtion, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(RightPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)

describeBy(TotalFixationCounter ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$TotalFixationCounter, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$TotalFixationCounter, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(TotalFixationCounter ~ ValidityScore, data = input_stage_1)

describeBy(SaccadeCounter ~ ValidityScore, data = input_stage_1)
kruskal.test(input_stage_1$SaccadeCounter, input_stage_1$ValidityScore)
pairwise.wilcox.test(input_stage_1$SaccadeCounter, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
boxplot(SaccadeCounter ~ ValidityScore, data = input_stage_1)


