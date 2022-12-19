using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using VRSTK.Scripts.TestControl;
//using VRQuestionnaireToolkit;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            /// <summary>
            /// PagesParameter class sets all statistic relevant parameter of a questionnaire, like:
            /// soscisurvey.de/help/doku.php/en:results:variables
            /// </summary>
            public class PagesParameters : MonoBehaviour
            {
                /// INTERVIEW IDENTIFICATION
                /// <summary>
                /// Time when the participant started the interview.
                /// </summary>
                [SerializeField]
                private string _STARTED = "";

                public string STARTED
                {
                    get { return _STARTED; }
                    set { _STARTED = value; }
                }

                /// Interview Progress variables
                /// <summary>
                /// Time when the participant most recently clicked the “next”
                /// </summary>
                [SerializeField]
                private string _LASTDATA = "";

                public string LASTDATA
                {
                    get { return _LASTDATA; }
                    set { _LASTDATA = value; }
                }

                /// <summary>
                /// The page most recently answered (and sent via Next) by the participant
                /// </summary>
                [SerializeField]
                private int _LASTPAGE = 0;
                public int LASTPAGE
                {
                    get { return _LASTPAGE; }
                    set { _LASTPAGE = value; }
                }

                /// <summary>
                /// The greatest number of any page answered by the participant.
                /// </summary>
                [SerializeField]
                private int _MAXPAGE = 0;

                public int MAXPAGE
                {
                    get { return _MAXPAGE; }
                    set { _MAXPAGE = value; }
                }

                /// <summary>
                ///  Did the participant reach the goodbye page (1) or not (0).
                /// </summary>
                [SerializeField]
                private int _FINISHED = 0;
                public int FINISHED
                {
                    get { return _FINISHED; }
                    set { _FINISHED = value; }
                }

                /// COMPILITION TIMES
                /// <summary>
                /// The sum of dwell times (in seconds) after correction for breaks.
                /// </summary>
                [SerializeField]
                private float _TIME_SUM = 0f;
                public float TIME_SUM
                {
                    get { return _TIME_SUM; }
                    set { _TIME_SUM = value; }
                }

                /// QUALITY INDICATOR
                /// <summary>
                /// An index that indicates how much faster a participant has completed the questionnaire than the typical participant (median) has done.
                /// </summary>
                [SerializeField]
                private int _TIME_RSI = 0;

                public int TIME_RSI
                {
                    get { return _TIME_RSI; }
                    set { _TIME_RSI = value; }
                }

                /// <summary>
                /// The percentage of answers omitted by the participant (0 to 100). 
                /// Only such questions and items are counted that have been shown to the participant – therefore someone dropping out early may have answered all questions (to this page, 0% missing)
                /// </summary>
                [SerializeField]
                private float _MISSING = 0f;

                public float MISSING
                {
                    get { return _MISSING; }
                    set { _MISSING = value; }
                }

                /// <summary>
                /// Negative points for extremely fast completion. This value is normed in such way that values of more than 100 points indicate low-quality data.
                /// If you prefer a more strict filtering, a threshold of 75 or even 50 points may as well be useful as a threshold of 200 for more liberal filtering.
                /// </summary>
                [SerializeField]
                private int _DEG_TIME = 0;

                public int  DEG_TIME
                {
                    get { return _DEG_TIME; }
                    set { _DEG_TIME = value; }
                }

                /// <summary>
                /// _DEG_TIME threshold to indicate low-quality data amd extremely fast completion.
                /// </summary>
                [SerializeField]
                private int _degTimeThreshold = 100;

                public int DegTimeThreshold
                {
                    get { return _degTimeThreshold; }
                    set { _degTimeThreshold = value; }
                }

                /// <summary>
                /// _DEG_TIME threshold to indicate low-quality data amd extremely fast completion.
                /// </summary>
                [SerializeField]
                private bool _degTimeLowQuality = false;

                public bool DegTimeLowQuality
                {
                    get { return _degTimeLowQuality; }
                    set { _degTimeLowQuality = value; }
                }

                /// <summary>
                /// _DEG_TIME threshold for one question in seconds.
                /// </summary>
                [SerializeField]
                private float _degTimeThresholdForOnePage = 15f;

                public float DegTimeThresholdForOnePage
                {
                    get { return _degTimeThresholdForOnePage; }
                    set { _degTimeThresholdForOnePage = value; }
                }

                /// <summary>
                /// _DEG_TIME value one question.
                /// </summary>
                [SerializeField]
                private int _degTimeValueForOnePage = 20;

                public int DegTimeValueForOnePage
                {
                    get { return _degTimeValueForOnePage; }
                    set { _degTimeValueForOnePage = value; }
                }

                /// <summary>
                /// 
                /// </summary>
                [SerializeField]
                private List<QuestionsAnswerRecord> _questionsAnswerRecordList;

                public List<QuestionsAnswerRecord> QuestionsAnswerRecordList
                {
                    get { return _questionsAnswerRecordList; }
                    set { _questionsAnswerRecordList = value; }
                }

                [SerializeField]
                private string _parametersAsMessage;
                
                public string ParametersAsMessage
                {
                    get { return _parametersAsMessage; }
                    set { _parametersAsMessage = value; }
                }

                private bool finisched = false;

                // Start is called before the first frame update
                void Start()
                {
                    if (TestStage.GetStarted())
                    {
                        _STARTED = System.DateTime.Now.ToString();

                        _questionsAnswerRecordList = new List<QuestionsAnswerRecord>();
                    }
                }

                // Update is called once per frame
                void Update()
                {
                    if (TestStage.GetStarted() && !finisched)
                    {
                        PageFactory pageFactory = GetComponent<PageFactory>();

                        if (pageFactory && (pageFactory.NumPages - 1) == pageFactory.CurrentPage)
                        {
                            //int questionsCounter = 0;
                            //int answeresCounter = 0;

                            for (int i = 1; i < pageFactory.NumPages - 1; i++)
                            {
                                if (i != (pageFactory.NumPages - 1))
                                {
                                    GameObject page = pageFactory.PageList[i];

                                    page.SetActive(true);

                                    PageParameters pageParameters = page.GetComponent<PageParameters>();

                                    // _TIME_SUM
                                    _TIME_SUM += pageParameters.TIME_nnn;

                                    // _DEG_TIME
                                    if (pageParameters.TIME_nnn < _degTimeThresholdForOnePage)
                                        _DEG_TIME += _degTimeValueForOnePage;

                                    // _MISSING
                                    // Q_Main.childCount
                                    for (int j = 0; j < page.transform.GetChild(0).GetChild(1).childCount; j++)
                                    {
                                        //questionsCounter++;
                                        _questionsAnswerRecordList.Add(new QuestionsAnswerRecord("", 1, 0));

                                        string childName = page.transform.GetChild(0).GetChild(1).GetChild(j).name;

                                        int currentQuestionsAnswerRecordListIndex = _questionsAnswerRecordList.Count - 1;//(i - 1) + j;

                                        Debug.Log("Page: " + i.ToString() + "-" + childName);

                                        if (childName.Contains("radioHorizontal_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.Radio radio = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>();
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + radio.QuestionnaireId + "-" + radio.QId;// + "-" + radio.QText;

                                            for (int k = 0; k < radio.RadioList.Count; k++)
                                                if (radio.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }

                                            if (answered)
                                            {
                                                _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;
                                                //answeresCounter++;
                                            }
                                        }
                                        else if (childName.Contains("radioGrid_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.RadioGrid>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.RadioGrid radioGrid = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.RadioGrid>();
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + radioGrid.QuestionnaireId + "-" + radioGrid.QId;// + "-" + radioGrid.QText;

                                            for (int k = 0; k < radioGrid.RadioList.Count; k++)
                                                if (radioGrid.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }

                                            if (answered)
                                            {
                                                _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;
                                                //answeresCounter++;
                                            }
                                        }
                                        else if (childName.Contains("checkbox_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Checkbox>() != null)
                                        {
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + "checkbox";
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;

                                            //answeresCounter++;
                                        }
                                        else if (childName.Contains("linearSlider_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Slider>() != null)
                                        {
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + "linearSlider";
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;

                                            //answeresCounter++;
                                        }
                                        else if (childName.Contains("dropdown_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Dropdown>() != null)
                                        {
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + "dropdown";
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;

                                            //answeresCounter++;
                                        }
                                        else if (childName.Contains("linearGrid_"))// && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.LinearGrid>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.LinearGrid linearGrid = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.LinearGrid>();
                                            _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].Question = i + "_" + childName + "_" + linearGrid.QuestionnaireId + "-" + linearGrid.QId; //+ "-" + linearGrid.QText;

                                            for (int k = 0; k < linearGrid.LinearGridList.Count; k++)
                                                if (linearGrid.LinearGridList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }

                                            if (answered)
                                            {
                                                _questionsAnswerRecordList[currentQuestionsAnswerRecordListIndex].AnsweredCounter = 1;
                                                //answeresCounter++;
                                            }
                                        }
                                    }

                                    page.SetActive(false);
                                }
                            }

                            finisched = true;

                            if (_DEG_TIME >= _degTimeThreshold)
                                _degTimeLowQuality = true;
                            else
                                _degTimeLowQuality = false;

                            GenerateQuestionnaire generateQuestionnaire = transform.parent.parent.GetComponent<GenerateQuestionnaire>();
                            QualityParameters qualityParameters = transform.parent.parent.GetComponent<QualityParameters>();
                            qualityParameters.TIME_SUM_s[generateQuestionnaire.Questionnaires.IndexOf(transform.parent.gameObject)] = _TIME_SUM;
                            qualityParameters.TIME_SUM_Message = _TIME_SUM.ToString();
                            qualityParameters.TIME_RSI_s[generateQuestionnaire.Questionnaires.IndexOf(transform.parent.gameObject)] = _TIME_SUM;
                            qualityParameters.DEG_TIME_s[generateQuestionnaire.Questionnaires.IndexOf(transform.parent.gameObject)] = _DEG_TIME;
                            qualityParameters.DEG_TIME_Message = _DEG_TIME.ToString();
                            qualityParameters.LastQuestionnaire_TIME_SUM = _TIME_SUM;

                            // _MISSING
                            int questionsCounter = _questionsAnswerRecordList.Count;
                            int answeresCounter = 0;

                            for (int i = 0; i < _questionsAnswerRecordList.Count; i++)//foreach (QuestionsAnswerRecord qar in _questionsAnswerRecordList)
                            {
                                // _MISSING
                                answeresCounter += _questionsAnswerRecordList[i].AnsweredCounter;

                                // MISSREL
                                if ((qualityParameters.QuestionsAnswerRecords.Count == _questionsAnswerRecordList.Count)) //&& (qualityParameters.QuestionsAnswerRecords[i].Question == _questionsAnswerRecordList[i].Question))
                                {
                                    qualityParameters.QuestionsAnswerRecords[i].AnsweredCounter += _questionsAnswerRecordList[i].AnsweredCounter;
                                    qualityParameters.QuestionsAnswerRecords[i].QuestionedCounter += _questionsAnswerRecordList[i].QuestionedCounter;
                                }
                                else
                                {
                                    bool exists = false;
                                    for (int j = 0; j < qualityParameters.QuestionsAnswerRecords.Count; j++)
                                        if (qualityParameters.QuestionsAnswerRecords[j].Question.Equals(_questionsAnswerRecordList[i].Question))
                                        {
                                            exists = true;
                                            break;
                                        }

                                    if (!exists)
                                        qualityParameters.QuestionsAnswerRecords.Add(new QuestionsAnswerRecord(_questionsAnswerRecordList[i].Question, _questionsAnswerRecordList[i].QuestionedCounter, _questionsAnswerRecordList[i].AnsweredCounter));
                                }
                            }
                            //Debug.Log(qualityParameters.QuestionsAnswerRecords.Count.ToString());
                            qualityParameters.EvaluateMissRels();

                            if (questionsCounter > 0)
                                _MISSING = (float)((float)((questionsCounter) - answeresCounter) / (float)(questionsCounter)) * 100;
                            else
                                _MISSING = 0f;

                            qualityParameters.MISSING_s[generateQuestionnaire.Questionnaires.IndexOf(transform.parent.gameObject)] = _MISSING;
                            qualityParameters.MISSING_Message = _MISSING.ToString();

                            PageParameters tempPageParameters = pageFactory.PageList[pageFactory.CurrentPage].GetComponent<PageParameters>();

                            // Message  structure: STARTED; LASTDATA; LASTPAGE; MAXPAGE; FINISHED; TIME_SUM; TIME_RSI; MISSING; DEG_TIME;
                            // DegTimeThreshold; DegTimeLowQuality; DegTimeThresholdForOnePage; DegTimeValueForOnePage; StandardDeviationStraightLineAnswer; AbsoluteDerivationOfResponseValue
                            _parametersAsMessage = string.Format("QualityParameters: {0}; {1}; {2}; {3}; {4}; {5}; {6}; {7}; {8}; {9}; {10}; {11}; {12}; {13}; {14}; {15};", _STARTED, _LASTDATA, _LASTPAGE, _MAXPAGE,
                                _FINISHED, _TIME_SUM, _TIME_RSI, _MISSING, _DEG_TIME, _degTimeThreshold, _degTimeLowQuality, _degTimeThresholdForOnePage, _degTimeValueForOnePage,
                                pageFactory.CurrentPage, tempPageParameters.StandardDeviationStraightLineAnswer, tempPageParameters.AbsoluteDerivationOfResponseValue);
                        }
                        else
                        {
                            if (pageFactory.CurrentPage > 0)
                            {
                                PageParameters tempPageParameters = pageFactory.PageList[pageFactory.CurrentPage].GetComponent<PageParameters>();

                                // Message  structure: STARTED; LASTDATA; LASTPAGE; MAXPAGE; FINISHED; TIME_SUM; TIME_RSI; MISSING; DEG_TIME;
                                // DegTimeThreshold; DegTimeLowQuality; DegTimeThresholdForOnePage; DegTimeValueForOnePage; CurrentPageNumber; StandardDeviationStraightLineAnswer;
                                // AbsoluteDerivationOfResponseValue
                                _parametersAsMessage = string.Format("QualityParameters: {0}; {1}; {2}; {3}; {4}; {5}; {6}; {7}; {8}; {9}; {10}; {11}; {12}; {13}; {14}; {15};", _STARTED, _LASTDATA, _LASTPAGE, _MAXPAGE,
                                    _FINISHED, _TIME_SUM, _TIME_RSI, _MISSING, _DEG_TIME, _degTimeThreshold, _degTimeLowQuality, _degTimeThresholdForOnePage, _degTimeValueForOnePage,
                                    pageFactory.CurrentPage, tempPageParameters.StandardDeviationStraightLineAnswer, tempPageParameters.AbsoluteDerivationOfResponseValue);
                            }
                        }
                    }
                }
            }
        }
    }
}
