using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class PageParameters : MonoBehaviour
            {
                //Completion Times
                /// <summary>
                /// The variables TIME001 etc. store the time (in seconds) that a participant stayed on a questionnaire page.
                /// </summary>
                [SerializeField]
                private float _TIME_nnn = 0f;

                public float TIME_nnn
                {
                    get { return _TIME_nnn; }
                    set { _TIME_nnn = value; }
                }

                [SerializeField]
                private float _current_Time_nnn = 0f;

                // First line checker
                [SerializeField]
                private float _standardDeviationStraightLineAnswer = -1f;        // straight line: 0; 1; 2; 3

                public float StandardDeviationStraightLineAnswer
                {
                    get { return _standardDeviationStraightLineAnswer; }
                    set { _standardDeviationStraightLineAnswer = value; }
                }

                // Second line checker
                //[SerializeField]
                //private float _patternAlgorithemStraightLineAnswer = 1f;        // close to zero -> straight line

                // Third line checker
                [SerializeField]
                private float _absoluteDerivationOfResponseValue = -1f;        // sensetive to straight, diagonal, and zigzag lines -> close to zero
                
                public float AbsoluteDerivationOfResponseValue
                {
                    get { return _absoluteDerivationOfResponseValue; }
                    set { _absoluteDerivationOfResponseValue = value; }
                }

                private float _started_time = 0f;
                private float _last_start_time = 0f;
                // Start is called before the first frame update
                void Start()
                {
                    _started_time = TestStage.GetTime();                    
                }

                // Update is called once per frame
                void Update()
                {
                    _current_Time_nnn = TestStage.GetTime() - _started_time;
                    if(_last_start_time != _started_time)
                        TIME_nnn += _current_Time_nnn;
                    else
                        TIME_nnn += Mathf.Abs(TIME_nnn - _current_Time_nnn);

                    CalculateStandardDeviationStraightLineAnswer();
                    CalculateAbsoluteDerivationOfResponseValue();

                    _last_start_time = _started_time;
                }

                public void CalculateStandardDeviationStraightLineAnswer()
                {
                    if (transform.GetChild(0).GetChild(1).childCount < 1)
                        return;

                    if (!gameObject.active) 
                        return;

                    float arithmeticMean = 0f;

                    float[] answers = new float[transform.GetChild(0).GetChild(1).childCount];
                    for (int j = 0; j < transform.GetChild(0).GetChild(1).childCount; j++)
                    {
                        string childName = transform.GetChild(0).GetChild(1).GetChild(j).name;
                        if (childName.Contains("radioHorizontal_"))
                        {
                            VRQuestionnaireToolkit.Radio radio = transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>();
                            for (int k = 0; k < radio.RadioList.Count; k++)
                                if (radio.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                {
                                    answers[j] = (k + 1);
                                    arithmeticMean += (k + 1);
                                    break;
                                }
                        }
                    }

                    if (answers.Length != 0)
                        arithmeticMean /= (float)answers.Length;
                    else
                        arithmeticMean = 0f;

                    float variance = 0f;
                    for (int i = 0; i < answers.Length; i++)
                        variance += Mathf.Pow(answers[i] - arithmeticMean, 2f);

                    float standardDeviation = -1f;
                    if (answers.Length != 0)
                        standardDeviation = variance / (float)(answers.Length - 1);
                    
                    if (standardDeviation != -1)
                        StandardDeviationStraightLineAnswer = Mathf.Sqrt(standardDeviation);
                    else
                        StandardDeviationStraightLineAnswer = standardDeviation;
                }

                public void CalculateAbsoluteDerivationOfResponseValue()
                {
                    if (transform.GetChild(0).GetChild(1).childCount < 2)
                        return;

                    if (!gameObject.active) 
                        return;

                    //int numberOfItems = transform.GetChild(0).GetChild(1).childCount;
                    float[] respones = new float[transform.GetChild(0).GetChild(1).childCount];

                    for (int j = 0; j < transform.GetChild(0).GetChild(1).childCount; j++)
                    {
                        string childName = transform.GetChild(0).GetChild(1).GetChild(j).name;
                        if (childName.Contains("radioHorizontal_"))
                        {
                            VRQuestionnaireToolkit.Radio radio = transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>();
                            for (int k = 0; k < radio.RadioList.Count; k++)
                                if (radio.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                {
                                    respones[j] = (k + 1);
                                    break;
                                }
                        }
                    }

                    float absoluteResponesValues = -1f;
                    for (int i = 0; i < respones.Length - 2; i++)
                        absoluteResponesValues += Mathf.Abs(respones[i+2] - (2 * respones[i + 1]) + respones[i]);

                    if ((respones.Length - 2) > 0)
                        AbsoluteDerivationOfResponseValue = (absoluteResponesValues + 1f) / (float)(respones.Length - 2);
                    else
                        AbsoluteDerivationOfResponseValue = -1f;
                }

                //public void CalculatePatternAlgorithemStraightLineAnswer()
                //{
                //    if (transform.GetChild(0).GetChild(1).childCount < 1)
                //        return;

                //}
            }
        }
    }
}
