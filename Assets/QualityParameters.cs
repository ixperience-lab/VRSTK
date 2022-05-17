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
                private float[] _TIME_RSI;

                public float[] TIME_RSI
                {
                    get { return _TIME_RSI; }
                    set { _TIME_RSI = value; }
                }

                /// <summary>
                /// Percentage of missing answers weighted by the other participants answering behavior.
                /// </summary>
                [SerializeField]
                private float _MISSREL = 0f;

                public float MISSREL
                {
                    get { return _MISSREL; }
                    set { _MISSREL = value; }
                }

                [SerializeField]
                private float[] _TIME_SUMs;

                public float[] TIME_SUMs
                {
                    get { return _TIME_SUMs; }
                    set
                    {
                        _TIME_SUMs = value;
                        EvaluateTimeRsi();
                    }
                }

                public void Start()
                {
                    _TIME_RSI = new float[_generateQuestionnaire.Questionnaires.Count];
                    TIME_SUMs = new float[_generateQuestionnaire.Questionnaires.Count];
                }

                private void EvaluateTimeRsi()
                {
                    float median = CalculateMedian();

                    for (int i = 0; i < TIME_SUMs.Length; i++)
                        if (TIME_SUMs[i] > 0)
                            TIME_RSI[i] = median / TIME_SUMs[i];
                        else
                            TIME_RSI[i] = 0f;
                }

                private float CalculateMedian()
                {
                    float median = 0f;
                    if (TIME_SUMs.Length == 1)
                        median = TIME_SUMs[0];
                    else if (TIME_SUMs.Length > 1)
                    {
                        int n = (TIME_SUMs.Length - 1) / 2;
                        int n1 = TIME_SUMs.Length / 2;

                        if ((TIME_SUMs.Length % 2) == 1)
                            median = 0.5f * (TIME_SUMs[n] + TIME_SUMs[n1]);
                        else
                            median = n1;
                    }
                    return median;
                }
            }
        }
    }
}
