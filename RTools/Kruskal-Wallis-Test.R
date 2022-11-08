# kruskal.test Feature tests tests to ValidityScore
# ===================================================
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
library(lsr)

kruskal_dir <- file.path("./kruskal_test", "")
if (!dir.exists(kruskal_dir)){
  dir.create(kruskal_dir)
}

# ===================================
# read data (baseline and task)
input_stage_0 <- read.csv2(file = './All_Participents_Stage0_DataFrame.csv')
input_stage_1 <- read.csv2(file = './All_Participents_Stage1_DataFrame.csv')

input_stage_1[input_stage_1$EvaluatedGlobalTIMERSICalc > 0, ]$EvaluatedGlobalTIMERSICalc <- 1
input_stage_1$ValidityScore <- 0
input_stage_1[(input_stage_1$EvaluatedGlobalTIMERSICalc > 0) & (input_stage_1$DegTimeLowQuality > 0), ]$ValidityScore <- 1

# eeg
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

temp <- input_stage_1
#================= 1 ============== 2 ============== 3 ============== 4 ============== 5 ============= 6 ============ 7 ============= 8 ============ 9 ============== 10 ============ 11 ============= 12 ============ 13 ============ 14 =====
temp$theta <- temp$AF3.theta + temp$F7.theta + temp$F3.theta + temp$FC5.theta + temp$T7.theta + temp$P7.theta + temp$O1.theta + temp$O2.theta + temp$P8.theta + temp$T8.theta + temp$AF4.theta + temp$F8.theta + temp$F4.theta + temp$FC6.theta
temp$alpha <- temp$AF3.alpha + temp$F7.alpha + temp$F3.alpha + temp$FC5.alpha + temp$T7.alpha + temp$P7.alpha + temp$O1.alpha + temp$O2.alpha + temp$P8.alpha + temp$T8.alpha + temp$AF4.alpha + temp$F8.alpha + temp$F4.alpha + temp$FC6.alpha 
temp$betaL <- temp$AF3.betaL + temp$F7.betaL + temp$F3.betaL + temp$FC5.betaL + temp$T7.betaL + temp$P7.betaL + temp$O1.betaL + temp$O2.betaL + temp$P8.betaL + temp$T8.betaL + temp$AF4.betaL + temp$F8.betaL + temp$F4.betaL + temp$FC6.betaL
temp$betaH <- temp$AF3.betaH + temp$F7.betaH + temp$F3.betaH + temp$FC5.betaH + temp$T7.betaH + temp$P7.betaH + temp$O1.betaH + temp$O2.betaH + temp$P8.betaH + temp$T8.betaH + temp$AF4.betaH + temp$F8.betaH + temp$F4.betaH + temp$FC6.betaH
temp$gamma <- temp$AF3.gamma + temp$F7.gamma + temp$F3.gamma + temp$FC5.gamma + temp$T7.gamma + temp$P7.gamma + temp$O1.gamma + temp$O2.gamma + temp$P8.gamma + temp$T8.gamma + temp$AF4.gamma + temp$F8.gamma + temp$F4.gamma + temp$FC6.gamma
input_stage_1 <- temp

kruskal_subfolder_dir <- file.path("./kruskal_test", "eeg")
if (!dir.exists(kruskal_subfolder_dir)){
  dir.create(kruskal_subfolder_dir)
}

# theta
descriptive_statistic_result <- describeBy(theta ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eeg_theta_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$theta ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eeg_theta_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$theta, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eeg_theta_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(theta ~ ValidityScore, data = input_stage_1)

shapiro.test(input_stage_1[3:5000, ]$theta)
hist(input_stage_1$theta)

# alpha
descriptive_statistic_result <- describeBy(alpha ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eeg_alpha_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$alpha ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eeg_alpha_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$alpha, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eeg_alpha_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(alpha ~ ValidityScore, data = input_stage_1)

# betaL
descriptive_statistic_result <- describeBy(betaL ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eeg_betaL_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$betaL ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eeg_betaL_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$betaL, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eeg_betaL_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(betaL ~ ValidityScore, data = input_stage_1)

# betaH
descriptive_statistic_result <- describeBy(betaH ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eeg_betaH_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$betaH ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eeg_betaH_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$betaH, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eeg_betaH_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(betaH ~ ValidityScore, data = input_stage_1)

# gamma
descriptive_statistic_result <- describeBy(gamma ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eeg_gamma_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$gamma ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eeg_gamma_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$gamma, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eeg_gamma_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(gamma ~ ValidityScore, data = input_stage_1)


# hrv 
# ------
kruskal_subfolder_dir <- file.path("./kruskal_test", "hrv")
if (!dir.exists(kruskal_subfolder_dir)){
  dir.create(kruskal_subfolder_dir)
}

descriptive_statistic_result <- describeBy(HeartRate ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_hrv_HeartRate_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$HeartRate ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_hrv_HeartRate_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$HeartRate, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_hrv_HeartRate_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(HeartRate ~ ValidityScore, data = input_stage_1)

# LFHFRatio
descriptive_statistic_result <- describeBy(LFHFRatio ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_hrv_LFHFRatio_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$LFHFRatio ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_hrv_LFHFRatio_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$LFHFRatio, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_hrv_LFHFRatio_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(LFHFRatio ~ ValidityScore, data = input_stage_1)

shapiro.test(input_stage_1[3:5000, ]$LFHFRatio)
hist(input_stage_1$LFHFRatio)

# SD1SD2Ratio
descriptive_statistic_result <- describeBy(SD1SD2Ratio ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_hrv_SD1SD2Ratio_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$SD1SD2Ratio ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_hrv_SD1SD2Ratio_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$SD1SD2Ratio, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_hrv_SD1SD2Ratio_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(SD1SD2Ratio ~ ValidityScore, data = input_stage_1)

# eda 
# ------
kruskal_subfolder_dir <- file.path("./kruskal_test", "eda")
if (!dir.exists(kruskal_subfolder_dir)){
  dir.create(kruskal_subfolder_dir)
}

# FilteredValueInMicroSiemens
descriptive_statistic_result <- describeBy(FilteredValueInMicroSiemens ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eda_FilteredValueInMicroSiemens_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$FilteredValueInMicroSiemens ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eda_FilteredValueInMicroSiemens_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$FilteredValueInMicroSiemens, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eda_FilteredValueInMicroSiemens_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(FilteredValueInMicroSiemens ~ ValidityScore, data = input_stage_1)

shapiro.test(input_stage_1[3:5000, ]$FilteredValueInMicroSiemens)
hist(input_stage_1$FilteredValueInMicroSiemens)

# eye
# ---------
kruskal_subfolder_dir <- file.path("./kruskal_test", "eye")
if (!dir.exists(kruskal_subfolder_dir)){
  dir.create(kruskal_subfolder_dir)
}

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

# LeftPercentChangePupilDialtion
descriptive_statistic_result <- describeBy(LeftPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eye_LeftPercentChangePupilDialtion_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$LeftPercentChangePupilDialtion ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eye_LeftPercentChangePupilDialtion_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$LeftPercentChangePupilDialtion, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eye_LeftPercentChangePupilDialtion_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(LeftPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)

shapiro.test(input_stage_1[3:5000, ]$LeftPercentChangePupilDialtion)
hist(input_stage_1$LeftPercentChangePupilDialtion)

# RightPercentChangePupilDialtion
descriptive_statistic_result <- describeBy(RightPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eye_RightPercentChangePupilDialtion_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$RightPercentChangePupilDialtion ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eye_RightPercentChangePupilDialtion_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$RightPercentChangePupilDialtion, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eye_RightPercentChangePupilDialtion_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(RightPercentChangePupilDialtion ~ ValidityScore, data = input_stage_1)

# TotalFixationCounter
descriptive_statistic_result <- describeBy(TotalFixationCounter ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eye_TotalFixationCounter_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$TotalFixationCounter ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eye_TotalFixationCounter_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$TotalFixationCounter, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eye_TotalFixationCounter_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(TotalFixationCounter ~ ValidityScore, data = input_stage_1)

# SaccadeCounter
descriptive_statistic_result <- describeBy(SaccadeCounter ~ ValidityScore, data = input_stage_1)
path_txt_file <- file.path(kruskal_subfolder_dir, "descriptive_statistic_eye_SaccadeCounter_validityscore_result.txt", "")
capture.output(descriptive_statistic_result, file = path_txt_file)
kruskal_summary <- kruskal.test(input_stage_1$SaccadeCounter ~ input_stage_1$ValidityScore)
path_txt_file <- file.path(kruskal_subfolder_dir, "kruskal_eye_SaccadeCounter_summary_results.txt", "")
capture.output(kruskal_summary, file = path_txt_file)
pairwise_wilcox <- pairwise.wilcox.test(input_stage_1$SaccadeCounter, input_stage_1$ValidityScore, paired = FALSE, p.adjust.method = "bonferroni")
path_txt_file <- file.path(t_test_subfolder_dir, "pairwise_wilcox_eye_SaccadeCounter_summary_results.txt", "")
capture.output(pairwise_wilcox, file = path_txt_file)
boxplot(SaccadeCounter ~ ValidityScore, data = input_stage_1)

shapiro.test(input_stage_1[3:5000, ]$SaccadeCounter)
hist(input_stage_1$SaccadeCounter)
