# Import the data and look at the first six rows

# Uncanny Valley Questionnaire (UVQ)
ImportUncannyValleyAnswers <- function()
{
  uncannyValleyAnswersRawData <- read.csv2(file = '../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/questionnaireID_exampleQE_ALL_UncannyValleyAnswers.csv')
  head(uncannyValleyAnswersRawData)
  return(uncannyValleyAnswersRawData);
}

# Simulation Sickness Questionnaire (SSQ)
ImportAllSSQAnswers <- function()
{
  allSSQAnswersRawData <- read.csv2(file = '../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/questionnaireID_SSQ_ALL_SSQAnswers.csv')
  head(allSSQAnswersRawData)
  return(allSSQAnswersRawData);
}

# Motion Sickness Susceptibility Questionnaire (MSSQ)
ImportAllMSSQAnswers <- function()
{
  allMSSQAnswersRawData <- read.csv2(file = '../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/MSSQAnswers.csv')
  head(allMSSQAnswersRawData)
  return(allMSSQAnswersRawData);
}

# uncannyValleyAnswersRawData <- ImportUncannyValleyAnswers()
# allSSQAnswersRawData <- ImportAllSSQAnswers()
# allMSSQAnswersRawData <- ImportAllMSSQAnswers()
