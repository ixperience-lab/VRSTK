library(psych)
library(stringi)
library(stringr)
library(jsonlite)
library(glue)
library(magrittr)
library(mnormt)

# Import the data and look at the first six rows

# Uncanny Valley Questionnaire (UVQ)
ImportUncannyValleyAnswers <- function(filePath)
{
  uncannyValleyAnswersRawData <- read.csv2(file = filePath) #'../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/questionnaireID_exampleQE_ALL_UncannyValleyAnswers.csv')
  head(uncannyValleyAnswersRawData)
  return(uncannyValleyAnswersRawData);
}

# Simulation Sickness Questionnaire (SSQ)
ImportAllSSQAnswers <- function(filePath)
{
  allSSQAnswersRawData <- read.csv2(file = filePath) #'../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/questionnaireID_SSQ_ALL_SSQAnswers.csv')
  head(allSSQAnswersRawData)
  return(allSSQAnswersRawData);
}

# Motion Sickness Susceptibility Questionnaire (MSSQ)
ImportAllMSSQAnswers <- function(filePath)
{
  allMSSQAnswersRawData <- read.csv2(file = filePath) #'../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/MSSQAnswers.csv')
  head(allMSSQAnswersRawData)
  return(allMSSQAnswersRawData);
}


condition <- 'Condition C'
condition_for_filename <- str_replace(condition, " ", "-")
type_q_answers <- 'Questionnaires/Answers'
q_1 <- '_ALL_UncannyValleyAnswers'
q_1_1 <- paste('_ALL_', condition_for_filename, '_UncannyValleyAnswers', sep='') 
q_2 <- '_ALL_SSQAnswers'
q_2_1 <- paste('_ALL_', condition_for_filename, '_SSQAnswers', sep='')
q_3 <- 'MSSQAnswers'
path <- file.path(condition,  type_q_answers, "/")

# 1 Import Questionnaire Answers
q_answers_files <- list.files(path, pattern=".csv", all.files=T, full.names=T)

for (file in q_answers_files) {
  # questionnaireID_exampleQE_ALL_UncannyValleyAnswers
  if(grepl(q_1, file)){
    uncannyValleyAnswersRawData <- ImportUncannyValleyAnswers(file)
    next
  }
  # questionnaireID_exampleQE_ALL_Condition-A_UncannyValleyAnswers
  if(grepl(q_1_1, file)){
    uncannyValleyConditionAnswersRawData <- ImportUncannyValleyAnswers(file)
    next
  }
  # questionnaireID_SSQ_ALL_SSQAnswers
  if(grepl(q_2, file)){
    SSQAnswersRawData <- ImportAllSSQAnswers(file)
    next
  }
  # questionnaireID_SSQ_ALL_Condtion-A_SSQAnswers
  if(grepl(q_2_1, file)){
    SSQConditionAnswersRawData <- ImportAllSSQAnswers(file)
    next
  }
  # questionnaireID_SSQ_ALL_SSQAnswers
  if(grepl(q_3, file)){
    MSSQAnswersRawData <- ImportAllMSSQAnswers(file)
    next
  }
}

# questionnaireID_exampleQE_ALL_UncannyValleyAnswers
# uncannyValleyAnswersRawData <- ImportUncannyValleyAnswers()

# questionnaireID_SSQ_ALL_SSQAnswers.csv
# allSSQAnswersRawData <- ImportAllSSQAnswers()

# MSSQAnswers
# allMSSQAnswersRawData <- ImportAllMSSQAnswers()

# 1. create descriptive statistics 
source("CreateDescreptiveStatisticOfQuestionnaireAnswers.r", echo=TRUE)


