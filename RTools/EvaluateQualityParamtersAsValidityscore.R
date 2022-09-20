# Evaluate Quality Paramters known as Validityscore
# - TIMERSI  = EvaluateTimeRsi()  # Eibezogen werden Alle Probanden einer Condition
# - MISSRELS = 0                  # Alle Fragen sind Pflichtfelder -> GEWICHTUNG aller Teilnemer ist 1 und MISSING 0 ->  MISSING * GEWICHTUNG -> 0 
# - DEGTIME  = _DEGTIME           # Aus PageQualityParameters rauslesen
# - MISSING  = 0                  # Alle Fragen sind Pflichtfelder -> 0
                                  # Es gab bis jetzt auch keine Unterbrechungen
# - Median                        # wird berechnet

EvaluateTimeRsi <- function()
{
  
  medianTimeRsi <- 0
  medianTimeRsi <- median(pagesTIMESUMsStage1$TIME_SUM)
  for(i in 1:nrow(pagesTIMESUMsStage1)){
    pagesTIMESUMsStage1$TIME_RSI[i] <- medianTimeRsi / pagesTIMESUMsStage1$TIME_SUM[i]
  }
  pagesTIMESUMsStage1$MEDIANForTRSI <- medianTimeRsi
  
  return(pagesTIMESUMsStage1)
}



#private void EvaluateTimeRsi()
#{
#  float[] copyOfTIME_SUM = new float[_generateQuestionnaire.Questionnaires.Count];
#  if (TIME_SUM_s.Length != copyOfTIME_SUM.Length) return;
#  
#  float median = CalculateMedian();
#   for (int i = 0; i < TIME_SUM_s.Length; i++)
#   {
#     if (TIME_SUM_s[i] > 0)
#       TIME_RSI_s[i] = median / TIME_SUM_s[i];
#     else
#       TIME_RSI_s[i] = 0f;
#     
#     TIME_RSI_Message += TIME_RSI_s[i].ToString() + "; ";
#   }
# }
# 
# private float CalculateMedian()
# {                    
#   float median = 0f;
#   float[] copyOfTIME_SUM = new float[_generateQuestionnaire.Questionnaires.Count];
#   
#   System.Array.Copy(TIME_SUM_s, copyOfTIME_SUM, _generateQuestionnaire.Questionnaires.Count);
#   System.Array.Sort(copyOfTIME_SUM);
#   
#   if (copyOfTIME_SUM.Length == 1)
#     median = copyOfTIME_SUM[0];
#   else if (copyOfTIME_SUM.Length > 1)
#   {
#     //float value = ((float)TIME_SUM_s.Length / 2.0f);
#     int n = (int)Mathf.Floor(((float)copyOfTIME_SUM.Length / 2.0f));
#     int n1 = (int) System.Math.Round(((float)copyOfTIME_SUM.Length / 2.0f), System.MidpointRounding.AwayFromZero);
#     
#     if ((copyOfTIME_SUM.Length % 2) == 0)
#       median = 0.5f * (copyOfTIME_SUM[n] + copyOfTIME_SUM[n1]);
#     else
#       median = copyOfTIME_SUM[n];
#   }
#   return median;
# }
# 
# public void EvaluateMissRels()
# {
#   int questionsCounter = _questionsAnswerRecords.Count;
#   
#   for(int i = 0; i < _generateQuestionnaire.Questionnaires.Count; i++)
#   {
#     GameObject go = _generateQuestionnaire.Questionnaires[i];
#     PagesParameters pp = go.transform.GetChild(0).GetComponent<PagesParameters>();
#     
#     float weightedAnsweresCounter = 0f;
#     if(pp != null)
#       for (int j = 0; j < pp.QuestionsAnswerRecordList.Count; j++)
#         weightedAnsweresCounter += (float) pp.QuestionsAnswerRecordList[j].AnsweredCounter * _questionsAnswerRecords[j].CalculateWeightFactor(); 
#         
#         _MISSREL_s[i] = (float) ((float)((questionsCounter - 1) - weightedAnsweresCounter) / (questionsCounter - 1));
#         
#         MISSREL_Message += _MISSREL_s[i].ToString() + "; ";
#   }
# }
# 
# public float CalculateWeightFactor()
# {
#   return _questionedCounter > 0 ? (float)((float)_answeredCounter / (float)_questionedCounter) : 0f ;
# }