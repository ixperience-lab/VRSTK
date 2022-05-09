using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            [System.Serializable]
            public class PageController : MonoBehaviour
            {
                [SerializeField]
                private GameObject _vrQuestionnaireFactory;
                [SerializeField]
                private PageFactory _pageFactory;
                [SerializeField]
                private GameObject _export;
                
                public List<GameObject> unansweredMandatoryQuestions;

                void Start()
                {
                    //init necessary relationships
                    _vrQuestionnaireFactory = GameObject.FindGameObjectWithTag("QuestionnaireFactory");
                    _export = GameObject.FindGameObjectWithTag("ExportToCSV");
                    _pageFactory = _vrQuestionnaireFactory.GetComponent<PageFactory>();
                    unansweredMandatoryQuestions = new List<GameObject>();
                }

                public bool CheckMandatoryQuestionsAnswered()
                {
                    int countMandatory = 0;
                    int answeredMandatory = 0;

                    for (int i = 0; i < _pageFactory.QuestionList.Count; i++)
                    {
                        if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Radio>() != null)
                        {
                            if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Radio>()
                                .QMandatory)
                            {
                                countMandatory++;

                                bool isAnswered = false;
                                for (int j = 0;
                                    j < _pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Radio>()
                                        .RadioList.Count;
                                    j++)
                                {
                                    if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[j].GetComponentInChildren<Toggle>().isOn)
                                    {
                                        isAnswered = true;
                                        answeredMandatory++;
                                    }
                                }
                                if (!isAnswered) // If this question is not answered yet.
                                {
                                    unansweredMandatoryQuestions.Add(_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].transform.parent.Find("QuestionText").gameObject);
                                }
                            }
                        }
                        else if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.RadioGrid>() != null)
                        {
                            if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.RadioGrid>()
                                .QMandatory)
                            {
                                countMandatory++;

                                bool isAnswered = false;
                                for (int j = 0;
                                    j < _pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0]
                                        .GetComponentInParent<VRQuestionnaireToolkit.RadioGrid>()
                                        .RadioList.Count;
                                    j++)
                                {
                                    if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[j].GetComponentInChildren<Toggle>()
                                        .isOn)
                                    {
                                        isAnswered = true;
                                        answeredMandatory++;
                                    }
                                }
                                if (!isAnswered) // If this question is not answered yet.
                                {
                                    unansweredMandatoryQuestions.Add(_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].transform.parent.Find("ConditionName").gameObject);
                                }
                            }
                        }
                        else if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Checkbox>() != null)
                        {
                            if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Checkbox>() != null)
                            {
                                if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Checkbox>()
                                    .QMandatory)
                                {
                                    countMandatory++;

                                    bool isAnswered = false;
                                    for (int j = 0;
                                        j < _pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.Checkbox>()
                                            .CheckboxList.Count;
                                        j++)
                                    {
                                        if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[j].GetComponentInChildren<Toggle>().isOn)
                                        {
                                            isAnswered = true;
                                            answeredMandatory++;
                                        }
                                    }
                                    if (!isAnswered) // If this question is not answered yet.
                                    {
                                        unansweredMandatoryQuestions.Add(_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].transform.parent.Find("QuestionText").gameObject);
                                    }
                                }
                            }
                        }
                        else if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.LinearGrid>() != null)
                        {
                            if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.LinearGrid>()
                                .QMandatory)
                            {
                                countMandatory++;

                                bool isAnswered = false;
                                for (int j = 0;
                                    j < _pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].GetComponentInParent<VRQuestionnaireToolkit.LinearGrid>()
                                        .LinearGridList.Count;
                                    j++)
                                {
                                    if (_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[j].GetComponentInChildren<Toggle>().isOn)
                                    {
                                        isAnswered = true;
                                        answeredMandatory++;
                                    }
                                }
                                if (!isAnswered) // If this question is not answered yet.
                                {
                                    unansweredMandatoryQuestions.Add(_pageFactory.GetComponent<PageFactory>().QuestionList[i].QuestionList[0].transform.parent.Find("QuestionText").gameObject);
                                }
                            }
                        }
                    }

                    if (countMandatory - answeredMandatory == 0)
                        return true;
                    else
                        return false;
                }

                /*
                 * GoToNextPage method
                 *
                 * Disables current page and enables next page
                 * If the user reaches the final page the "Next" button becomes the "Submit" button with the underlying functionality
                 * to call the Save() and Export() Method
                 * Verifies that all "mandatory" questions on this page have been answered
                 */
                public void GoToNextPage()
                {
                    if (CheckMandatoryQuestionsAnswered() || _pageFactory.CurrentPage == 0)
                    {
                        // _pageFactory.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false);
                        _pageFactory.PageList[_pageFactory.CurrentPage].SetActive(false);
                        ++_pageFactory.CurrentPage;
                        _pageFactory.PageList[_pageFactory.CurrentPage].SetActive(true);

                        //reached second-last page
                        if (_pageFactory.PageList.Count - 2 == _pageFactory.CurrentPage)
                        {
                            GameObject q_footer = GameObject.Find("Q_Footer");
                            TextMeshProUGUI nextButton = q_footer.GetComponentInChildren<TextMeshProUGUI>();
                            nextButton.text = "Submit";
                        }

                        if (_pageFactory.PageList.Count - 1 == _pageFactory.CurrentPage)
                        {
                            _export.GetComponent<ExportToCSV>().Save();
                        }
                    }
                    else
                    {
                        foreach (GameObject obj in unansweredMandatoryQuestions) // Make each unanswered question blink in red.
                        {
                            StartCoroutine(ChangeTextColor(obj));
                        }
                        unansweredMandatoryQuestions.Clear();
                        //   _pageFactory.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(true);
                    }
                }

                /*
                 * GoToPreviousPage method
                 *
                 * Disables current page and enables previous page
                 */
                public void GoToPreviousPage()
                {
                    _pageFactory.PageList[_pageFactory.CurrentPage].SetActive(false);
                    --_pageFactory.CurrentPage;
                    _pageFactory.PageList[_pageFactory.CurrentPage].SetActive(true);
                }

                IEnumerator ChangeTextColor(GameObject textObj)
                {
                    float increment = 0.05f;
                    float timer = 0;
                    while (timer <= 1)
                    {
                        timer += increment;
                        textObj.GetComponent<TextMeshProUGUI>().color =
                            Color.Lerp(Color.black, Color.red, Mathf.Abs(Mathf.Sin(3 * Mathf.PI * timer))); // blink 3 times from black to red.
                        yield return new WaitForSeconds(increment);
                    }
                }
            }
        }
    }
}
