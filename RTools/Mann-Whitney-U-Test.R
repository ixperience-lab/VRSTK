# Mann-Whitney-U-Test/Wilcoxon-Test Feature tests tests to ValidityScore
# ======================================================================
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
library(tidyr)
#install.packages("coin")
library("coin")
#install.packages("rstatix")
library("rstatix")
#install.packages("nortest")
library(nortest)
#install.packages("stargazer") #Use this to install it, do this only once
#library(stargazer)
library(data.table)



wilcoxon_dir <- file.path("./wilcoxon_test", "")
if (!dir.exists(wilcoxon_dir)){
  dir.create(wilcoxon_dir)
}

# ===================================
# read data (baseline and task)
input_stage_0 <- read.csv2(file = './All_Participents_Stage0_DataFrame.csv')
input_stage_1 <- read.csv2(file = './All_Participents_Stage1_DataFrame.csv')

input_stage_1[input_stage_1$EvaluatedGlobalTIMERSICalc > 0, ]$EvaluatedGlobalTIMERSICalc <- 1
input_stage_1$ValidityScore <- 0
input_stage_1$ValidityScore_Group <- "high"
input_stage_1[(input_stage_1$EvaluatedGlobalTIMERSICalc > 0) & (input_stage_1$DegTimeLowQuality > 0), ]$ValidityScore <- 1
input_stage_1[(input_stage_1$EvaluatedGlobalTIMERSICalc > 0) & (input_stage_1$DegTimeLowQuality > 0), ]$ValidityScore_Group <- "low"

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

wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "eeg")
if (!dir.exists(wilcoxon_subfolder_dir)){
  dir.create(wilcoxon_subfolder_dir)
}

for (test_variable in c("theta", "alpha", "betaL", "betaH", "gamma", "eng", "exc", "str", "rel", "int", "foc")){
  temp_column <- eval(call(name = "$", as.symbol("input_stage_1"), as.symbol(test_variable)))
  temp_df <- data.frame (t_variable  = temp_column, ValidityScore = input_stage_1$ValidityScore, ValidityScore_Group=input_stage_1$ValidityScore_Group)
  
  path_txt_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eeg_",paste0(test_variable, "_results.txt")))
  sink(path_txt_file)
  
  cat("======================================================================================================================================================\n")
  cat(paste("EEG Variance test between test_variable: ", test_variable, " and Group: ValidityScore" ,"\n", sep=""))
  cat("======================================================================================================================================================\n")
  
  print(ad.test(temp_df$t_variable))
  
  print(describeBy(t_variable ~ ValidityScore, data = temp_df))
  
  print(wilcox.test(temp_df$t_variable ~ temp_df$ValidityScore, exact = FALSE, correct = FALSE, conf.int = FALSE))
  
  print(temp_df %>%  wilcox_effsize(t_variable ~ ValidityScore))
  
  sink()
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eeg_",paste0(test_variable, "_hist_plot.png")))
  ggplot(temp_df, aes(t_variable)) + 
    geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", outlier.shape = NA) +
    scale_x_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) + 
    labs(title=paste0(test_variable, " histogram plot"))  +
    xlab(label=test_variable) +
    ylab(label="density") +
    geom_density(alpha=0.2, color='red', fill='red', adjust=1, outlier.shape = NA)  +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(limits = quantile(temp_df$t_variable, c(0.1, 0.9)))
  ggsave(path_png_file, width = 15, height = 10)
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eeg_",paste0(test_variable, "_boxplot.png")))
  ggplot(temp_df, aes(x=ValidityScore, y=t_variable, fill=ValidityScore_Group, group=ValidityScore)) + 
    stat_boxplot(geom = "errorbar", width = 0.25) +
    geom_boxplot(outlier.shape = NA, outlier.size= NA, width = 0.5) + 
    labs(title=paste0(test_variable, " box plot")) +
    xlab(label="ValidityScore") +
    ylab(label=test_variable) +
    theme(legend.position="none") +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_y_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) 
  ggsave(path_png_file, width = 15, height = 10)
  
  #png(file=path_png_file, width=1500, height=1000)
  #h <- hist(temp_df$t_variable, main="", xlab="", ylab="", prob = TRUE, col="blue", cex.axis=1.5)
  #grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
  #hist(temp_df$t_variable, main="", xlab="", ylab="", prob = TRUE, col="blue", cex.axis=1.5, add = TRUE)
  #mtext(side=1, line=2.5, test_variable, cex=2)
  #mtext(side=2, line=2.5, "density", cex=2)
  #mtext(side=3, line=0.5, paste0(test_variable, "-Value Histogram"), cex=3) 
  #lines(density(temp_df$t_variable), lwd = 2, col = "red")
  #dev.off()
  
  #path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eeg_",paste0(test_variable, "_boxplot.png")))
  #png(file=path_png_file, width=1500, height=1000)
  #b <- boxplot(t_variable ~ ValidityScore, data = temp_df,  xlab="", ylab="", outline=FALSE)
  #grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
  #boxplot(t_variable ~ ValidityScore, data = temp_df,   xlab="", ylab="", outline=FALSE)
  #mtext(side=1, line=2.5, "ValidityScore", cex=2)
  #mtext(side=2, line=2.5, test_variable, cex=2)
  #mtext(side=3, line=0.5, paste0(test_variable, "-Value Boxplot"), cex=3) 
  #dev.off()
}


# hrv 
# ------
wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "hrv")
if (!dir.exists(wilcoxon_subfolder_dir)){
  dir.create(wilcoxon_subfolder_dir)
}

for (test_variable in c("HeartRate", "RPeaks", "RRI", "RRMin", "RRMean", "RRMax", "SDSD", "SD1", "SD2", "SDNN", "RMSSD",
                        "SD1SD2Ratio", "VLFPeak", "LFPeak", "HFPeak", "VLFAbs", "LFAbs", "HFAbs", "VLFLog", "LFLog", "HFLog", "LFNorm", 
                        "HFNorm", "LFHFRatio", "FBTotal")){
  temp_column <- eval(call(name = "$", as.symbol("input_stage_1"), as.symbol(test_variable)))
  temp_df <- data.frame (t_variable  = temp_column, ValidityScore = input_stage_1$ValidityScore, ValidityScore_Group=input_stage_1$ValidityScore_Group)
  
  
  path_txt_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_hrv_",paste0(test_variable, "_results.txt")))
  sink(path_txt_file)
  
  cat("======================================================================================================================================================\n")
  cat(paste("HRV Variance test between test_variable: ", test_variable, " and Group: ValidityScore" ,"\n", sep=""))
  cat("======================================================================================================================================================\n")
  
  print(ad.test(temp_df$t_variable))
  
  print(describeBy(t_variable ~ ValidityScore, data = temp_df))
  
  print(wilcox.test(temp_df$t_variable ~ temp_df$ValidityScore, exact = FALSE, correct = FALSE, conf.int = FALSE))
  
  print(temp_df %>%  wilcox_effsize(t_variable ~ ValidityScore))
  
  sink()
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_hrv_",paste0(test_variable, "_hist_plot.png")))
  ggplot(temp_df, aes(t_variable)) + 
    geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", outlier.shape = NA) +
    #scale_x_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) + 
    labs(title=paste0(test_variable, " histogram plot"))  +
    xlab(label=test_variable) +
    ylab(label="density") +
    geom_density(alpha=0.2, color='red', fill='red', adjust=1, outlier.shape = NA)  +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_x_continuous(limits = quantile(itemp_df$t_variable, c(0.1, 0.9)))
  ggsave(path_png_file, width = 15, height = 10)
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_hrv_",paste0(test_variable, "_boxplot.png")))
  ggplot(temp_df, aes(x=ValidityScore, y=t_variable, fill=ValidityScore_Group, group=ValidityScore)) + 
    stat_boxplot(geom = "errorbar", width = 0.25) +
    geom_boxplot(outlier.shape = NA, outlier.size= NA, width = 0.5) + 
    labs(title=paste0(test_variable, " box plot")) +
    xlab(label="ValidityScore") +
    ylab(label=test_variable) +
    theme(legend.position="none") +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_y_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) 
  ggsave(path_png_file, width = 15, height = 10)
  
}


# test
# ------
#wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "hrv")
#if (!dir.exists(wilcoxon_subfolder_dir)){
#  dir.create(wilcoxon_subfolder_dir)
#}
#path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_hrv_",paste0(test_variable, "_Test_________box_plot.png")))
#ggplot(input_stage_1, aes(x=ValidityScore, y=theta, fill=ValidityScore_Group, group=ValidityScore)) + 
#  stat_boxplot(geom = "errorbar", width = 0.25) +
#  geom_boxplot(outlier.shape = NA, outlier.size= NA, width = 0.5) + 
#  labs(title="theta histogram plot") +
#  xlab(label="test_variable") +
#  ylab(label="density___AAA") +
#  theme(legend.position="none") +
#  theme(plot.title = element_text(hjust = 0.5)) +
#  scale_y_continuous(limits = quantile(input_stage_1$theta, c(0.01, 0.99))) 
#ggsave(path_png_file, width = 15, height = 10)

#path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_hrv_",paste0(test_variable, "_Test_________hist_plot.png")))
#ggplot(input_stage_1, aes(theta)) + 
#  geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", outlier.shape = NA) +
#  scale_x_continuous(limits = quantile(input_stage_1$theta, c(0.01, 0.99))) + 
#  labs(title="theta histogram plot") +
#  xlab(label="test_variable") +
#  ylab(label="density___AAA") +
#  geom_density(alpha=0.2, color='red', fill='red', adjust=1, outlier.shape = NA)  +
#  theme(plot.title = element_text(hjust = 0.5)) +
#  scale_x_continuous(limits = quantile(input_stage_1$theta, c(0.1, 0.9)))
#ggsave(path_png_file, width = 15, height = 10)


# eda 
# ------
wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "eda")
if (!dir.exists(wilcoxon_subfolder_dir)){
  dir.create(wilcoxon_subfolder_dir)
}

for (test_variable in c("FilteredValueInMicroSiemens", "onsets", "peaks", "amps", "RawValueInMicroSiemens")){
  temp_column <- eval(call(name = "$", as.symbol("input_stage_1"), as.symbol(test_variable)))
  temp_df <- data.frame (t_variable  = temp_column, ValidityScore = input_stage_1$ValidityScore, ValidityScore_Group=input_stage_1$ValidityScore_Group)
  
  
  path_txt_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eda_",paste0(test_variable, "_results.txt")))
  sink(path_txt_file)
  
  cat("======================================================================================================================================================\n")
  cat(paste("EDA Variance test between test_variable: ", test_variable, " and Group: ValidityScore" ,"\n", sep=""))
  cat("======================================================================================================================================================\n")
  
  print(ad.test(temp_df$t_variable))
  
  print(describeBy(t_variable ~ ValidityScore, data = temp_df))
  
  print(wilcox.test(temp_df$t_variable ~ temp_df$ValidityScore, exact = FALSE, correct = FALSE, conf.int = FALSE))
  
  print(temp_df %>%  wilcox_effsize(t_variable ~ ValidityScore))
  
  sink()
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eda_",paste0(test_variable, "_hist_plot.png")))
  ggplot(temp_df, aes(t_variable)) + 
    geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", outlier.shape = NA) +
    #scale_x_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) + 
    labs(title=paste0(test_variable, " histogram plot"))  +
    xlab(label=test_variable) +
    ylab(label="density") +
    geom_density(alpha=0.2, color='red', fill='red', adjust=1, outlier.shape = NA)  +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_x_continuous(limits = quantile(itemp_df$t_variable, c(0.1, 0.9)))
  ggsave(path_png_file, width = 15, height = 10)
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eda_",paste0(test_variable, "_boxplot.png")))
  ggplot(temp_df, aes(x=ValidityScore, y=t_variable, fill=ValidityScore_Group, group=ValidityScore)) + 
    stat_boxplot(geom = "errorbar", width = 0.25) +
    geom_boxplot(outlier.shape = NA, outlier.size= NA, width = 0.5) + 
    labs(title=paste0(test_variable, " box plot")) +
    xlab(label="ValidityScore") +
    ylab(label=test_variable) +
    theme(legend.position="none") +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_y_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) 
  ggsave(path_png_file, width = 15, height = 10)
  
}


# eye
# ---------
wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "eye")
if (!dir.exists(wilcoxon_subfolder_dir)){
  dir.create(wilcoxon_subfolder_dir)
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

for (test_variable in c("TotalFixationCounter", "SaccadeCounter", "LeftPercentChangePupilDialtion", "RightPercentChangePupilDialtion", "LeftPupilDiameter", 
                        "RightPupilDiameter", "FixationCounter", "FixationDuration", "MeasuredVelocity", "SaccadesDiffX", "SaccadesMeanX", "SaccadesSdX", 
                        "SaccadesMinX", "SaccadesMaxX", "CognitiveActivityLeftPupilDiamter", "CognitiveActivityLeftPupilDiamter", "CognitiveActivityRightPupilDiamter", 
                        "LeftMeanPupilDiameter", "LeftPupilDiameterDifferenceToMean", "RightMeanPupilDiameter", "RightPupilDiameterDifferenceToMean")){
  temp_column <- eval(call(name = "$", as.symbol("input_stage_1"), as.symbol(test_variable)))
  temp_df <- data.frame (t_variable  = temp_column, ValidityScore = input_stage_1$ValidityScore, ValidityScore_Group=input_stage_1$ValidityScore_Group)
  
  path_txt_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eye_",paste0(test_variable, "_results.txt")))
  sink(path_txt_file)
  
  cat("======================================================================================================================================================\n")
  cat(paste("EYE Variance test between test_variable: ", test_variable, " and Group: ValidityScore" ,"\n", sep=""))
  cat("======================================================================================================================================================\n")
  
  print(ad.test(temp_df$t_variable))
  
  print(describeBy(t_variable ~ ValidityScore, data = temp_df))
  
  print(wilcox.test(temp_df$t_variable ~ temp_df$ValidityScore, exact = FALSE, correct = FALSE, conf.int = FALSE))
  
  print(temp_df %>%  wilcox_effsize(t_variable ~ ValidityScore))
  
  sink()
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eye_",paste0(test_variable, "_hist_plot.png")))
  ggplot(temp_df, aes(t_variable)) + 
    geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", outlier.shape = NA) +
    #scale_x_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) + 
    labs(title=paste0(test_variable, " histogram plot"))  +
    xlab(label=test_variable) +
    ylab(label="density") +
    geom_density(alpha=0.2, color='red', fill='red', adjust=1, outlier.shape = NA)  +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_x_continuous(limits = quantile(itemp_df$t_variable, c(0.1, 0.9)))
  ggsave(path_png_file, width = 15, height = 10)
  
  path_png_file <- file.path(wilcoxon_subfolder_dir, paste0("wilcoxon_eye_",paste0(test_variable, "_box_plot.png")))
  ggplot(temp_df, aes(x=ValidityScore, y=t_variable, fill=ValidityScore_Group, group=ValidityScore)) + 
    stat_boxplot(geom = "errorbar", width = 0.25) +
    geom_boxplot(outlier.shape = NA, outlier.size= NA, width = 0.5) + 
    labs(title=paste0(test_variable, " box plot")) +
    xlab(label="ValidityScore") +
    ylab(label=test_variable) +
    theme(legend.position="none") +
    theme(plot.title = element_text(hjust = 0.5)) #+
    #scale_y_continuous(limits = quantile(temp_df$t_variable, c(0.01, 0.99))) 
  ggsave(path_png_file, width = 15, height = 10)
  
}

# Test version
# ===================================
# read data 
input_ssq_ca <- read.csv2(file = './Condition A/RResults/Questionnaires/AllSSQConditionStatisticResults_DataFrame.csv')
input_ssq_cb <- read.csv2(file = './Condition B/RResults/Questionnaires/AllSSQConditionStatisticResults_DataFrame.csv')
input_ssq_cc <- read.csv2(file = './Condition C/RResults/Questionnaires/AllSSQConditionStatisticResults_DataFrame.csv')

# ssq
# ---------
wilcoxon_subfolder_dir <- file.path("./wilcoxon_test", "ssq")
if (!dir.exists(wilcoxon_subfolder_dir)){
  dir.create(wilcoxon_subfolder_dir)
}

path_txt_file <- file.path(wilcoxon_subfolder_dir, "wilcoxon_ssq_results.txt")
sink(path_txt_file)

cat("======================================================================================================================================================\n")
cat("SSQ Variance test between Conditions: A and B \n")
cat("======================================================================================================================================================\n")

print(ad.test(input_ssq_ca$mean))
print(ad.test(input_ssq_cb$mean))

print(wilcox.test(input_ssq_ca$mean, input_ssq_cb$mean, exact = FALSE, correct = FALSE, conf.int = FALSE))

sink()

#xlim('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16') + 
ggplot(input_ssq_ca, aes(x=mean)) + 
  geom_histogram() #+ scale_x_continuous(breaks=seq(0,1.5,0.1), limits=c(0,1.5))

path_png_file <- file.path(wilcoxon_subfolder_dir, "wilcoxon_ssq_ca_hist_plot.png")
ggplot(input_ssq_ca, aes(mean)) + 
  geom_histogram(aes(y=..density..), color='gray50', stat = "bin", fill='blue', alpha=0.2, position = "identity", bins = 8) +
  labs(title="Condition A histogram")  +
  xlab(label="Condition A") +
  ylab(label="Density") +
  geom_density(alpha=0.2, color='red', fill='red', adjust=1)  +
  theme(plot.title = element_text(hjust = 0.5)) #+
ggsave(path_png_file, width = 15, height = 10)

path_png_file <- file.path(wilcoxon_subfolder_dir, "wilcoxon_ssq_cb_hist_plot.png")
ggplot(input_ssq_cb, aes(mean)) + 
  geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", bins = 8) +
  labs(title="Condition B histogram")  +
  xlab(label="Condition B") +
  ylab(label="Density") +
  geom_density(alpha=0.2, color='red', fill='red', adjust=1)  +
  theme(plot.title = element_text(hjust = 0.5)) #+
ggsave(path_png_file, width = 15, height = 10)

path_png_file <- file.path(wilcoxon_subfolder_dir, "wilcoxon_ssq_cb_hist_plot.png")
ggplot(input_ssq_cc, aes(mean)) + 
  geom_histogram(aes(y=..density..), color='gray50', fill='blue', alpha=0.2, position = "identity", bins = 8) +
  labs(title="Condition B histogram")  +
  xlab(label="Condition B") +
  ylab(label="Density") +
  geom_density(alpha=0.2, color='red', fill='red', adjust=1)  +
  theme(plot.title = element_text(hjust = 0.5)) #+
ggsave(path_png_file, width = 15, height = 10)
