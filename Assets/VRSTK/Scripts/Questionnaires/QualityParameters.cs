using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class QualityParameters : MonoBehaviour
            {
                public GenerateQuestionnaire _generateQuestionnaire;

                /// COMPILITION TIMES
                /// <summary>
                /// The sum of dwell times (in seconds) after correction for breaks. 
                /// </summary>
                [SerializeField]
                private float _lastQuestionnaire_TIME_SUM = 0f;
                public float LastQuestionnaire_TIME_SUM
                {
                    get { return _lastQuestionnaire_TIME_SUM; }
                    set 
                    {
                        _lastQuestionnaire_TIME_SUM = value;
                        EvaluateTimeRsi();
                    }
                }

                /// QUALITY INDICATOR
                /// <summary>
                /// An index that indicates how much faster a participant has completed the questionnaire than the typical participant (median) has done.
                /// Values above 1 identify faster respondents, values below 1 slower respondents.
                /// RSI (named “relative speed index”)
                /// </summary>
                [SerializeField]
                private float[] _TIME_RSI_s;

                public float[] TIME_RSI_s
                {
                    get { return _TIME_RSI_s; }
                    set { _TIME_RSI_s = value; }
                }

                [SerializeField]
                private string _TIME_RSI_Message = "";

                public string TIME_RSI_Message
                {
                    get { return _TIME_RSI_Message; }
                    set { _TIME_RSI_Message = value; }
                }

                /// <summary>
                /// Percentage of missing answers weighted by the other participants answering behavior.
                /// </summary>
                [SerializeField]
                private float[] _MISSREL_s;

                public float[] MISSREL_s
                {
                    get { return _MISSREL_s; }
                    set { _MISSREL_s = value; }
                }

                [SerializeField]
                private string _MISSREL_Message = "";

                public string MISSREL_Message
                {
                    get { return _MISSREL_Message; }
                    set { _MISSREL_Message = value; }
                }

                /// <summary>
                /// The percentage of answers omitted by the participant (0 to 100). 
                /// Only such questions and items are counted that have been shown to the participant – therefore someone dropping out early may have answered all questions (to this page, 0% missing)
                /// </summary>
                [SerializeField]
                private float [] _MISSING_s;

                public float[] MISSING_s
                {
                    get { return _MISSING_s; }
                    set { _MISSING_s = value; }
                }

                [SerializeField]
                private string _MISSING_Message = "";

                public string MISSING_Message
                {
                    get { return _MISSING_Message; }
                    set { _MISSING_Message = value; }
                }

                /// <summary>
                /// ...
                /// </summary>
                [SerializeField]
                private float[] _TIME_SUM_s;

                public float[] TIME_SUM_s
                {
                    get { return _TIME_SUM_s; }
                    set { _TIME_SUM_s = value; }
                }

                [SerializeField]
                private string _TIME_SUM_Message = "";

                public string TIME_SUM_Message
                {
                    get { return _TIME_SUM_Message; }
                    set { _TIME_SUM_Message = value; }
                }

                /// <summary>
                /// ...
                /// </summary>
                [SerializeField]
                private List<QuestionsAnswerRecord> _questionsAnswerRecords;

                public List<QuestionsAnswerRecord> QuestionsAnswerRecords
                {
                    get { return _questionsAnswerRecords; }
                    set { _questionsAnswerRecords = value; }
                }

                /// <summary>
                /// Negative points for extremely fast completion. This value is normed in such way that values of more than 100 points indicate low-quality data.
                /// If you prefer a more strict filtering, a threshold of 75 or even 50 points may as well be useful as a threshold of 200 for more liberal filtering.
                /// </summary>
                [SerializeField]
                private int[] _DEG_TIME_s;

                public int[] DEG_TIME_s
                {
                    get { return _DEG_TIME_s; }
                    set { _DEG_TIME_s = value; }
                }

                [SerializeField]
                private string _DEG_TIME_Message = "";

                public string DEG_TIME_Message
                {
                    get { return _DEG_TIME_Message; }
                    set { _DEG_TIME_Message = value; }
                }

                public void Start()
                {
                    _TIME_RSI_s = new float[_generateQuestionnaire.Questionnaires.Count];
                    _TIME_SUM_s = new float[_generateQuestionnaire.Questionnaires.Count];
                    _MISSREL_s = new float[_generateQuestionnaire.Questionnaires.Count];
                    _MISSING_s = new float[_generateQuestionnaire.Questionnaires.Count];
                    _DEG_TIME_s = new int[_generateQuestionnaire.Questionnaires.Count];
                    _questionsAnswerRecords = new List<QuestionsAnswerRecord>();
                }

                private void EvaluateTimeRsi()
                {
                    float[] copyOfTIME_SUM = new float[_generateQuestionnaire.Questionnaires.Count];
                    if (TIME_SUM_s.Length != copyOfTIME_SUM.Length) return;

                    float median = CalculateMedian();

                    for (int i = 0; i < TIME_SUM_s.Length; i++)
                    {
                        if (TIME_SUM_s[i] > 0)
                            TIME_RSI_s[i] = median / TIME_SUM_s[i];
                        else
                            TIME_RSI_s[i] = 0f;

                        TIME_RSI_Message += TIME_RSI_s[i].ToString() + "; ";
                    }
                }

                private float CalculateMedian()
                {                    
                    float median = 0f;
                    float[] copyOfTIME_SUM = new float[_generateQuestionnaire.Questionnaires.Count];
                    
                    System.Array.Copy(TIME_SUM_s, copyOfTIME_SUM, _generateQuestionnaire.Questionnaires.Count);
                    System.Array.Sort(copyOfTIME_SUM);

                    if (copyOfTIME_SUM.Length == 1)
                        median = copyOfTIME_SUM[0];
                    else if (copyOfTIME_SUM.Length > 1)
                    {
                        //float value = ((float)TIME_SUM_s.Length / 2.0f);
                        int n = (int)Mathf.Floor(((float)copyOfTIME_SUM.Length / 2.0f));
                        int n1 = (int) System.Math.Round(((float)copyOfTIME_SUM.Length / 2.0f), System.MidpointRounding.AwayFromZero);

                        if ((copyOfTIME_SUM.Length % 2) == 0)
                            median = 0.5f * (copyOfTIME_SUM[n] + copyOfTIME_SUM[n1]);
                        else
                            median = copyOfTIME_SUM[n];
                    }
                    return median;
                }

                public void EvaluateMissRels()
                {
                    int questionsCounter = _questionsAnswerRecords.Count;

                    for(int i = 0; i < _generateQuestionnaire.Questionnaires.Count; i++)
                    {
                        GameObject go = _generateQuestionnaire.Questionnaires[i];
                        PagesParameters pp = go.transform.GetChild(0).GetComponent<PagesParameters>();
                        
                        float weightedAnsweresCounter = 0f;
                        if(pp != null)
                            for (int j = 0; j < pp.QuestionsAnswerRecordList.Count; j++)
                                weightedAnsweresCounter += (float) pp.QuestionsAnswerRecordList[j].AnsweredCounter * _questionsAnswerRecords[j].CalculateWeightFactor(); 
                        
                        _MISSREL_s[i] = (float) ((float)((questionsCounter - 1) - weightedAnsweresCounter) / (questionsCounter - 1));

                        MISSREL_Message += _MISSREL_s[i].ToString() + "; ";
                    }
                }
            }
        }
    }
}
