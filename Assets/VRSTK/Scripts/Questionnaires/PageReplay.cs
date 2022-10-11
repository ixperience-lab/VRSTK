using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class PageReplay : MonoBehaviour
            {   
                [SerializeField]
                private System.Int32 _currentActivePageIndex;

                public System.Int32 CurrentActivePageIndex
                {
                    get 
                    { 
                        return _currentActivePageIndex; 
                    }
                    set
                    {
                        _currentActivePageIndex = value;
                        Replay();
                    }
                }

                [SerializeField]
                private System.String _selectedContentToggle;

                public System.String SelectedContentToggle
                {
                    get 
                    { 
                        return _selectedContentToggle; 
                    }
                    set 
                    {
                        _selectedContentToggle = value; 
                    }
                }

                [SerializeField]
                public GameObject _cameraTracking;

                private System.Int32 _lastCurrentActivePageIndex = -1;
                private System.String _lastSelectedContentToggle = "";

                private void Replay()
                {
                    PageFactory pf = GetComponent<PageFactory>();

                    if (!pf.gameObject.active) return;

                    GameObject page = pf.PageList[_currentActivePageIndex];

                    if (_lastCurrentActivePageIndex != _currentActivePageIndex && _lastCurrentActivePageIndex > -1)
                    {
                        LineRenderer lineRenderer = _cameraTracking.GetComponent<LineRenderer>();
                        lineRenderer.positionCount = 0;

                        GameObject[] spheres = GameObject.FindGameObjectsWithTag("ReplaySphere_");
                        if(spheres != null && spheres.Length != 0)
                            for (int i = 0; i < spheres.Length; i++)
                                GameObject.DestroyImmediate(spheres[i]);                            

                        pf.PageList[_lastCurrentActivePageIndex].SetActive(false);
                    }

                    page.SetActive(true);

                    // Trying to render RectTransform in pause mode of unity app
                    LayoutRebuilder.ForceRebuildLayoutImmediate(page.GetComponent<RectTransform>());
                    LayoutRebuilder.ForceRebuildLayoutImmediate(page.GetComponent<RectTransform>());
                    
                    page.GetComponent<Canvas>().enabled = true;
                    
                    // Q_Header
                    page.transform.GetChild(0).GetChild(0).gameObject.GetComponent<Image>().enabled = true;
                    // Q_Header.DescribtionText
                    page.transform.GetChild(0).GetChild(0).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = true;
                    // Q_Header.DescribtionText.Mandatory
                    page.transform.GetChild(0).GetChild(0).GetChild(0).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = true;

                    // Q_Main
                    page.transform.GetChild(0).GetChild(1).gameObject.GetComponent<Image>().enabled = true;
                                                
                    if (_currentActivePageIndex != 0 && _currentActivePageIndex != (pf.PageList.Count - 1))
                    {
                        if (_selectedContentToggle != "" && _lastSelectedContentToggle != _selectedContentToggle)
                        {
                            string[] tempLoop = _selectedContentToggle.Split(';');
                            for (int i = 0; i < tempLoop.Length-1; i++)
                            {
                                // [First/ Final / RadioHorizontel_ / Checkbox_ / LinearSlider_ / LinearGrid_ / DropDown_].[Text/ Radio_ / Checkbox_ / Slider_ / LinearGrid_ / DropDownX].[null / Radio / Checkbox / Slider / Redio / DropDown].[null/IsOn /IsOn    /Value /IsOn /Text    ]
                                string[] tempContent = tempLoop[i].Split('.');
                                // [First/ Final / RadioHorizontel_ / Checkbox_ / LinearSlider_ / LinearGrid_ / DropDown_]
                                string root = tempContent[0];
                                // [Text/ Radio_ / Checkbox_ / Slider_ / LinearGrid_ / DropDownX]
                                string root_leaf = tempContent[1];
                                // [null / Radio / Checkbox / Slider / Redio / DropDown]
                                string root_leaf_leaf = tempContent[2];
                                // [null / Radio / Checkbox / Slider / Redio / DropDown] Type
                                string root_leaf_leaf_type = tempContent[3];
                                // [null / Radio / Checkbox / Slider / Redio / DropDown] Value
                                string root_leaf_leaf_value = tempContent[4];

                                Debug.Log(_currentActivePageIndex.ToString() + "___" + root + "_" + root_leaf + "_" + root_leaf_leaf + "_" + root_leaf_leaf_type + "_" + root_leaf_leaf_value);

                                if (page.transform.GetChild(0).GetChild(1).Find(root) == null) continue;

                                GameObject root_child = page.transform.GetChild(0).GetChild(1).Find(root).gameObject;

                                if (root_child == null) continue;

                                if (root_child.transform.Find(root_leaf) == null) continue;

                                GameObject root_leaf_child = root_child.transform.Find(root_leaf).gameObject;

                                if (root_leaf_child.transform.Find(root_leaf_leaf) == null) continue;

                                GameObject root_leaf_leaf_child = root_leaf_child.transform.Find(root_leaf_leaf).gameObject;

                                if (root_leaf_leaf_child != null)
                                {
                                    if (root_leaf_leaf_type == "Toggle")
                                    {
                                        bool value = root_leaf_leaf_value == "True" ? true : false;
                                        root_leaf_leaf_child.transform.gameObject.GetComponent<Toggle>().isOn = value;
                                    }

                                    if (root_leaf_leaf_type == "Slider")
                                    {
                                        // Debug.Log(root_leaf_leaf_child.name + " value: " + root_leaf_leaf_value);
                                        // Debug.Log(root_leaf_leaf_child.transform.GetChild(0).gameObject.name + " value: " + root_leaf_leaf_value);
                                        float value = float.Parse(root_leaf_leaf_value);
                                        root_leaf_leaf_child.GetComponent<Slider>().value = value;
                                    }

                                    if (root_leaf_leaf_type == "TMP_Dropdown")
                                    {
                                        int value = int.Parse(root_leaf_leaf_value);
                                        root_leaf_leaf_child.GetComponent<TMP_Dropdown>().value = value;
                                        root_leaf_leaf_child.GetComponent<TMP_Dropdown>().Select();
                                    }
                                }
                            }
                        }
                    }
                    else
                        page.transform.GetChild(0).GetChild(1).GetChild(0).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = true;

                    // Q_Footer
                    page.transform.GetChild(0).GetChild(2).gameObject.GetComponent<Image>().enabled = true;
                    // Q_Futter.ButtonNext
                    page.transform.GetChild(0).GetChild(2).GetChild(0).gameObject.GetComponent<Image>().enabled = true;
                    // Q_Futter.ButtonNext.Next
                    page.transform.GetChild(0).GetChild(2).GetChild(0).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = true;

                    if (_currentActivePageIndex != 0 && _currentActivePageIndex != (pf.PageList.Count - 1))
                    {
                        page.transform.GetChild(0).GetChild(2).GetChild(1).gameObject.SetActive(true);
                        // Q_Futter.ButtonPrevious
                        page.transform.GetChild(0).GetChild(2).GetChild(1).gameObject.GetComponent<Image>().enabled = true;
                        // Q_Futter.ButtonPrevious.Previous
                        page.transform.GetChild(0).GetChild(2).GetChild(1).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = true;
                    }
                    else
                    {
                        page.transform.GetChild(0).GetChild(2).GetChild(1).gameObject.SetActive(false);
                        // Q_Futter.ButtonPrevious
                        page.transform.GetChild(0).GetChild(2).GetChild(1).gameObject.GetComponent<Image>().enabled = false;
                        // Q_Futter.ButtonPrevious.Previous
                        page.transform.GetChild(0).GetChild(2).GetChild(1).GetChild(0).gameObject.GetComponent<TMPro.TextMeshProUGUI>().enabled = false;
                    }

                    _lastCurrentActivePageIndex = _currentActivePageIndex;
                    _lastSelectedContentToggle = _selectedContentToggle;

                    // Trying to render RectTransform in pause mode of unity app
                    LayoutRebuilder.ForceRebuildLayoutImmediate(page.GetComponent<RectTransform>());
                    LayoutRebuilder.ForceRebuildLayoutImmediate(page.GetComponent<RectTransform>());
                }
            }
        }
    }
}
