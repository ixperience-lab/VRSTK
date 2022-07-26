# Import the data and look at the first six rows
ImportUncannyValleyAnswers <- function()
{
  uncannyValleyAnswersRawData <- read.csv2(file = '../Assets/VRSTK/SampleSceneData/Questionnaires/Answers/questionnaireID_exampleQE_ALL_UncannyValleyAnswers.csv')
  head(uncannyValleyAnswersRawData)
  return(uncannyValleyAnswersRawData);
}

