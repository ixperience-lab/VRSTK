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

                /// <summary>
                /// ...
                /// </summary>
                [SerializeField]
                private float[] _TIME_SUM_s;

                public float[] TIME_SUM_s
                {
                    get { return _TIME_SUM_s; }
                    set
                    {
                        _TIME_SUM_s = value;
                        EvaluateTimeRsi();
                    }
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
                    float median = CalculateMedian();

                    for (int i = 0; i < TIME_SUM_s.Length; i++)
                        if (TIME_SUM_s[i] > 0)
                            TIME_RSI_s[i] = median / TIME_SUM_s[i];
                        else
                            TIME_RSI_s[i] = 0f;
                }

                private float CalculateMedian()
                {
                    float median = 0f;
                    if (TIME_SUM_s.Length == 1)
                        median = TIME_SUM_s[0];
                    else if (TIME_SUM_s.Length > 1)
                    {
                        int n = (TIME_SUM_s.Length - 1) / 2;
                        int n1 = TIME_SUM_s.Length / 2;

                        if ((TIME_SUM_s.Length % 2) == 1)
                            median = 0.5f * (TIME_SUM_s[n] + TIME_SUM_s[n1]);
                        else
                            median = n1;
                    }
                    return median;
                }

                public void EvaluateMissRels()
                {
                    int questionsCounter = _questionsAnswerRecords.Count;

                    for(int i = 0; i < _generateQuestionnaire.Questionnaires.Count; i++)
                    {
                        GameObject go = _generateQuestionnaire.Questionnaires[i];
                        PagesParameters pp = go.transform.GetChild(0).GetChild(0).GetComponent<PagesParameters>();
                        //Debug.Log(pp);
                        //Debug.Log(pp.QuestionsAnswerRecordList.Count);
                        float weightedAnsweresCounter = 0f;
                        if(pp != null)
                            for (int j = 0; j < pp.QuestionsAnswerRecordList.Count; j++)
                                weightedAnsweresCounter += pp.QuestionsAnswerRecordList[j].AnsweredCounter * _questionsAnswerRecords[j].CalculateWeightFactor(); 
                        
                        _MISSREL_s[i] = (questionsCounter - weightedAnsweresCounter) / questionsCounter;
                    }
                }
            }
        }
    }
}
