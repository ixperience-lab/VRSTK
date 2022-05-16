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
            public class PagesParameters : MonoBehaviour
            {
                // Interview Identification
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

                // Interview Progress variables
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


                // Quality indicator
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

                private bool _checkboxPageUnanswered = false;


                // Start is called before the first frame update
                void Start()
                {
                    if (TestStage.GetStarted())
                    {
                        _STARTED = System.DateTime.Now.ToString();
                    }
                }

                // Update is called once per frame
                void Update()
                {
                    if (TestStage.GetStarted())
                    {
                        PageFactory pageFactory = GetComponent<PageFactory>();
                        
                        if (pageFactory && (pageFactory.NumPages - 1) == pageFactory.CurrentPage)
                        {
                            int questionsCounter = 0;
                            int answeresCounter = 0;
                            
                            for (int i = 0; i < pageFactory.NumPages - 1; i++)
                            {
                                if (i != 0 && i != (pageFactory.NumPages - 2))
                                {
                                    GameObject page = pageFactory.PageList[i];
                                    // Q_Main.childCount
                                    for(int j = 0; j < page.transform.GetChild(0).GetChild(1).childCount; j++)
                                    {
                                        questionsCounter++;
                                        if (page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("radioHorizontal_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.Radio radio = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Radio>();
                                            for (int k = 0; k < radio.RadioList.Count; k++)
                                                if (radio.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }
                                            
                                            if(answered)
                                                answeresCounter++;
                                        }
                                        else if (page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("radioGrid_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.RadioGrid>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.RadioGrid radioGrid = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.RadioGrid>();
                                            for (int k = 0; k < radioGrid.RadioList.Count; k++)
                                                if (radioGrid.RadioList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }

                                            if (answered)
                                                answeresCounter++;
                                        }
                                        else if (page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("checkbox_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Checkbox>() != null)
                                        {
                                            answeresCounter++;
                                        }
                                        else if (page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("linearSlider_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Slider>() != null)
                                        {
                                            answeresCounter++;
                                        }
                                        else if (page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("dropdown_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.Dropdown>() != null)
                                        {
                                            answeresCounter++;
                                        }
                                        else //if(page.transform.GetChild(0).GetChild(1).GetChild(j).name.Contains("linearGrid_") && page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.LinearGrid>() != null)
                                        {
                                            bool answered = false;
                                            VRQuestionnaireToolkit.LinearGrid linearGrid = page.transform.GetChild(0).GetChild(1).GetChild(j).GetComponent<VRQuestionnaireToolkit.LinearGrid>();
                                            for (int k = 0; k < linearGrid.LinearGridList.Count; k++)
                                                if (linearGrid.LinearGridList[k].transform.GetChild(0).GetComponent<Toggle>().isOn)
                                                {
                                                    answered = true;
                                                    break;
                                                }

                                            if (answered)
                                                answeresCounter++;
                                        }
                                    }
                                }
                            }

                            if (questionsCounter > 0)
                                _MISSING = (questionsCounter - answeresCounter) / questionsCounter;
                            else
                                _MISSING = 0f;
                        }
                    }
                }
            }
        }
    }
}
