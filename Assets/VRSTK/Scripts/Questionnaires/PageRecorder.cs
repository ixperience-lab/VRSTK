using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class PageRecorder : MonoBehaviour
            {
                public Telemetry.Event TrackPages;

                void Start()
                {
                }

                void Update()
                {
                    if (TestStage.GetStarted())
                    {
                        GetComponent<EventSender>().SetEventValue("CurrentActivePageIndex_PageReplay", (System.Int32)transform.GetChild(0).GetComponent<PageFactory>().CurrentPage);

                        string selectedContentToggle_PageReplay = "";
                        PageFactory pf = transform.GetChild(0).GetComponent<PageFactory>();

                        GameObject page = pf.PageList[pf.CurrentPage];
                        //page.Q_Panel.Q_Main.[Text/RadioHorizontel_/Checkbox_/LinearSlider_/LinearGrid_/DropDown]
                        for(int i = 0; i < page.transform.GetChild(0).GetChild(1).childCount; i++)
                        {   
                            GameObject child = page.transform.GetChild(0).GetChild(1).GetChild(i).gameObject;
                            for (int j = 0; j < child.transform.childCount; j++)
                            {
                                //[First/Final/RadioHorizontel_/Checkbox_/LinearSlider_/LinearGrid_/DropDown_].[Text/Radio_/Checkbox_/Slider_/LinearGrid_/DropDownX]
                                if (child.transform.GetChild(j).childCount > 0)
                                {
                                    //[Text/Radio_/Checkbox_/Slider_/LinearGrid_/DropDownX].[null/Radio/Checkbox/Slider/Redio/DropDown]
                                    //                                                      [null/IsOn /IsOn    /Value /IsOn /Text    ]
                                    if (child.transform.GetChild(j).GetChild(0).gameObject.GetComponent<Toggle>())
                                    {
                                        selectedContentToggle_PageReplay += child.name + "." + child.transform.GetChild(j).gameObject.name + "." + child.transform.GetChild(j).GetChild(0).gameObject.name + "." + "Toggle" + "." + child.transform.GetChild(j).GetChild(0).gameObject.GetComponent<Toggle>().isOn + ";";
                                    }
                                    else if (child.transform.GetChild(j).GetChild(0).gameObject.GetComponent<Slider>() != null)
                                    {
                                        selectedContentToggle_PageReplay += child.name + "." + child.transform.GetChild(j).gameObject.name + "." + child.transform.GetChild(j).GetChild(0).gameObject.name + "." + "Slider" + "." + child.transform.GetChild(j).GetChild(0).gameObject.GetComponent<Slider>().value + ";";
                                    }
                                    else if (child.transform.GetChild(j).GetChild(0).gameObject.GetComponent<TMP_Dropdown>() != null)
                                    {
                                        selectedContentToggle_PageReplay += child.name + "." + child.transform.GetChild(j).gameObject.name + "." + child.transform.GetChild(j).GetChild(0).gameObject.name + "." + "TMP_Dropdown" + "." + child.transform.GetChild(j).GetChild(0).GetComponent<TMPro.TMP_Dropdown>().value + ";";
                                    }
                                }
                            }
                        }

                        GetComponent<EventSender>().SetEventValue("SelectedContentToggle_PageReplay", selectedContentToggle_PageReplay);

                        GetComponent<EventSender>().Deploy();
                    }
                }
            }
        }
    }
}
