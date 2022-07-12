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
                private float _standardDeviationStraightLineAnswer = 1f;        // close to zero -> straight line

                // Second line checker

                // Third line checker
                [SerializeField]
                private float _absoluteDerivationOfResponseValue = 0f;        // sensetive to straight, diagonal, and zigzag lines


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

                    _last_start_time = _started_time;
                }

                public void CalculateStandardDeviationStraightLineAnswer()
                {
                    float[] answers = new float[transform.GetChild(0).GetChild(1).childCount];
                    float arithmeticMean = 4f;
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
                                    break;
                                }
                        }
                    }

                    float variance = 0f;
                    for (int i = 0; i < answers.Length; i++)
                        variance += Mathf.Pow(answers[i] - arithmeticMean, 2f);

                    if (answers.Length != 0)
                        variance /= (float)answers.Length;
                    else
                        variance = 0f;

                    if (variance != 0)
                        _standardDeviationStraightLineAnswer = Mathf.Sqrt(variance);
                    else
                        _standardDeviationStraightLineAnswer = 1f;
                }

                public void CalculateAbsoluteDerivationOfResponseValue()
                {
                    if (transform.GetChild(0).GetChild(1).childCount < 2)
                        return;

                    int numberOfItems = transform.GetChild(0).GetChild(1).childCount;
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

                    float absoluteResponesValues = 0f;
                    for (int i = 0; i < respones.Length - 2; i++)
                        absoluteResponesValues += Mathf.Abs(respones[i+2] - 2 * respones[i + 1] + respones[i]);

                    if ((respones.Length - 2) != 0)
                        _absoluteDerivationOfResponseValue = absoluteResponesValues / (respones.Length - 2);
                    else
                        _absoluteDerivationOfResponseValue = 0f;
                }
            }
        }
    }
}
