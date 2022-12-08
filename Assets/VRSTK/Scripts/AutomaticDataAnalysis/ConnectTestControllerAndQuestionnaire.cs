using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.TestControl;
using VRQuestionnaireToolkit;
using UnityEngine.UI;

public class ConnectTestControllerAndQuestionnaire : MonoBehaviour
{
    public GameObject _testController;
    private bool _setStartParameter = false;
    // Start is called before the first frame update
    void Start()
    {
        SetTestControllerStartParameterToQuestionnaireParameter();
    }

    // Update is called once per frame
    void Update()
    {
        if (!_setStartParameter)
            SetTestControllerStartParameterToQuestionnaireParameter();
    }

    private void SetTestControllerStartParameterToQuestionnaireParameter()
    {
        if(TestStage.GetStarted() && _testController != null)
        {
            TestController testController = _testController.GetComponent<TestController>();
            for (int i = 0; i < testController.testStages.Length; i++)
            {
                if (testController.testStages[i].active)
                {
                    TestStage testStage = testController.testStages[i].GetComponent<TestStage>();

                    for (int j = 0; j < testStage.startProperties.Length; j++)
                    {
                        if (testStage.startProperties[j].text.text.ToLower().Contains("id"))
                        {
                            GetComponent<StudySetup>().ParticipantId = testStage.startProperties[j].GetValue();
                            _setStartParameter = true;
                        }
                        if (testStage.startProperties[j].text.text.ToLower().Contains("condition") && testStage.startProperties[j].GetValue().ToLower().Equals("true"))
                        {
                            GetComponent<StudySetup>().Condition = testStage.startProperties[j].text.text;
                            _setStartParameter = true;
                        }
                    }
                    if (_setStartParameter)
                        break;
                }
            }
        }
    }
}
