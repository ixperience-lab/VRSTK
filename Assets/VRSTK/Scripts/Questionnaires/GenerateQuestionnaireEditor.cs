using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using SimpleJSON;
using System.IO;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class GenerateQuestionnaireEditor : EditorWindow
            {

                Vector2 _scrollPosition = Vector2.zero;
                float _widht;
                float _height;


                public GameObject _questionnaireToolkit;
                public GameObject _lastQuestionnaireToolkit;

                private string _jsonInputPath_1 = "Assets/Questionnaires/Data/Questions/questionnaire_example";
                private string _jsonInputPath_2 = "Assets/Questionnaires/Data/Questions/NASA_TLX";
                private string _jsonInputPath_3 = "Assets/Questionnaires/Data/Questions/SUS Presence Questionnaire";
                private string _jsonInputPath_4 = "Assets/Questionnaires/Data/Questions/Simulation Sickness Questionnaire";
                private string _jsonInputPath_5 = "Assets/Questionnaires/Data/Questions/IPQ Questionnaire";
                private string _jsonInputPath_6 = "Assets/Questionnaires/Data/Questions/System Usability Scale Questionnaire";
                private string _jsonInputPath_7 = "Assets/Questionnaires/Data/Questions/Virtual Embodiment Questionnaire";
                private string _jsonInputPath_8 = "";
                private string _jsonInputPath_9 = "";
                private string _jsonInputPath_10 = "";

                private List<string> _jsonInputFiles;

                public int _indexOfAQuestionnaireToSetActive = 0;

                private PageFactory _pageFactory;
                private ExportToCSV _exportToCsvScript;
                
                private JSONArray _qData;
                private JSONArray _qConditions;
                private JSONArray _qOptions;

                private GameObject currentQuestionnaire;
                private int numberQuestionnaires;
                private string qId;
                private string pId;

                [MenuItem("Window/VRSTK/Generate Questionnaire")]
                public static void ShowWindow()
                {
                    EditorWindow.GetWindow(typeof(GenerateQuestionnaireEditor));
                }

                void OnEnable()
                {
                    _questionnaireToolkit = Selection.activeGameObject;
                    if (_lastQuestionnaireToolkit == null && _questionnaireToolkit != null)
                    {                        
                        //_trackedComponents = new bool[_questionnaireToolkit.GetComponents(typeof(Component)).Length];
                        //_trackedVariables = new bool[_questionnaireToolkit.GetComponents(typeof(Component)).Length][];
                        //lastTrackedObject = trackedObject;
                    }
                }

                private void OnInspectorUpdate()
                {
                    _questionnaireToolkit = Selection.activeGameObject;
                    if (_questionnaireToolkit != _lastQuestionnaireToolkit)
                    {
                        Repaint();
                    }
                }

                private void OnGUI()
                {
                    if (_questionnaireToolkit != null && _questionnaireToolkit.name.Equals("VRQuestionnaireToolkit"))
                    {
                        _widht = position.width;
                        _height = position.height;

                        _scrollPosition = GUILayout.BeginScrollView(_scrollPosition, true, true, GUILayout.Width(_widht), GUILayout.Height(_height));

                        //GUILayout.ExpandWidth(false);

                        EditorGUILayout.LabelField("Set paths and variables to create QuestionnaireToolkit:");
                        EditorGUILayout.BeginVertical();
                        {
                            _jsonInputPath_1 = EditorGUILayout.TextField("JsonInputPath_1", _jsonInputPath_1);
                            _jsonInputPath_2 = EditorGUILayout.TextField("JsonInputPath_2", _jsonInputPath_2);
                            _jsonInputPath_3 = EditorGUILayout.TextField("JsonInputPath_3", _jsonInputPath_3);
                            _jsonInputPath_4 = EditorGUILayout.TextField("JsonInputPath_4", _jsonInputPath_4);
                            _jsonInputPath_5 = EditorGUILayout.TextField("JsonInputPath_5", _jsonInputPath_5);
                            _jsonInputPath_6 = EditorGUILayout.TextField("JsonInputPath_6", _jsonInputPath_6);
                            _jsonInputPath_7 = EditorGUILayout.TextField("JsonInputPath_7", _jsonInputPath_7);
                            _jsonInputPath_8 = EditorGUILayout.TextField("JsonInputPath_8", _jsonInputPath_8);
                            _jsonInputPath_9 = EditorGUILayout.TextField("JsonInputPath_9", _jsonInputPath_9);
                            _jsonInputPath_10 = EditorGUILayout.TextField("JsonInputPath_10", _jsonInputPath_10);
                        }
                        EditorGUILayout.EndVertical();
                        EditorGUILayout.BeginHorizontal();
                        {
                            EditorGUILayout.LabelField("IndexOfAQuestionnaireToSetActive");
                            _indexOfAQuestionnaireToSetActive = EditorGUILayout.IntField("", _indexOfAQuestionnaireToSetActive);
                        }
                        EditorGUILayout.EndHorizontal();
                        if (GUILayout.Button("Generate Questionnaire"))
                        {
                            GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                            if (q.Questionnaires.Count == 0)
                                GenerateQuestionnaire();
                        }
                        if (GUILayout.Button("Delete Questionnaire"))
                        {
                            GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                            if(q.Questionnaires != null && q.Questionnaires.Count > 0)
                            {
                                DeleteQuestionnaire(q);
                            }

                        }
                        GUILayout.EndScrollView();
                    }
                    else
                        EditorGUILayout.LabelField("Please select a GameObject to generate a questionnaire in the inspector.");
                    
                    _lastQuestionnaireToolkit = _questionnaireToolkit;
                }

                private void FireEvent()
                {
                    Debug.Log("QuestionnaireFinishedEvent");
                }

                public void PrepareConfigurations()
                {   
                    _exportToCsvScript = _questionnaireToolkit.GetComponentInChildren<ExportToCSV>();
                    _exportToCsvScript.QuestionnaireFinishedEvent.AddListener(FireEvent);

                    numberQuestionnaires = 1;
                    _jsonInputFiles = new List<string>();

                    if (_jsonInputPath_1 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_1);
                    }
                    if (_jsonInputPath_2 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_2);
                    }
                    if (_jsonInputPath_3 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_3);
                    }
                    if (_jsonInputPath_4 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_4);
                    }
                    if (_jsonInputPath_5 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_5);
                    }
                    if (_jsonInputPath_6 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_6);
                    }
                    if (_jsonInputPath_7 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_7);
                    }
                    if (_jsonInputPath_8 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_8);
                    }
                    if (_jsonInputPath_9 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_9);
                    }
                    if (_jsonInputPath_10 != "")
                    {
                        _jsonInputFiles.Add(_jsonInputPath_10);
                    }
                }

                public void InitializeQuestionnaire(GenerateQuestionnaire q)
                {
                    foreach (string InputPath in _jsonInputFiles)
                        GenerateNewQuestionnaire(InputPath, q);

                    for (int i = 1; i < q.Questionnaires.Count; i++)
                        q.Questionnaires[i].SetActive(false);

                    q.Questionnaires[_indexOfAQuestionnaireToSetActive].SetActive(true);
                }

                void DeleteQuestionnaire(GenerateQuestionnaire q)
                {
                    for (int i = 0; i < q.Questionnaires.Count; i++)
                    {
                        _questionnaireToolkit.transform.Find(q.Questionnaires[i].name).gameObject.SetActive(false);
                        DestroyImmediate(_questionnaireToolkit.transform.Find(q.Questionnaires[i].name).gameObject);
                    }
                    q.Questionnaires.Clear();
                }

                void GenerateNewQuestionnaire(string inputPath, GenerateQuestionnaire q)
                {
                    if (numberQuestionnaires > 1)
                        currentQuestionnaire.SetActive(false);

                    currentQuestionnaire = Instantiate(q.questionnaire);
                    currentQuestionnaire.name = "Questionnaire_" + numberQuestionnaires;



                    // Place in hierarchy 
                    RectTransform radioGridRec = currentQuestionnaire.GetComponent<RectTransform>();
                    radioGridRec.SetParent(q.QuestionRecTest);

                    Undo.RegisterCompleteObjectUndo(radioGridRec, "Added to Questionnaire objects");
                    Undo.FlushUndoRecordObjects();


                    radioGridRec.localPosition = new Vector3(0, 0, 0);
                    radioGridRec.localRotation = Quaternion.identity;
                    radioGridRec.localScale = new Vector3(radioGridRec.localScale.x * 0.01f, radioGridRec.localScale.y * 0.01f, radioGridRec.localScale.z * 0.01f);

                    _pageFactory = q.GetComponentInChildren<PageFactory>();

                    

                    q.Questionnaires.Add(currentQuestionnaire);
                    numberQuestionnaires++;

                    EditorUtility.SetDirty(radioGridRec);

                    AssetDatabase.Refresh();

                    AssetDatabase.SaveAssets();

                    ReadJson(inputPath);

                    //for (int i = 0; i < _pageFactory.PageList.Count; i++)
                    //{
                    //    for (int j = 0; j < _pageFactory.PageList[i].GetComponent<RectTransform>().childCount; j++)
                    //    {
                    //        Undo.RegisterCompleteObjectUndo(_pageFactory.PageList[i].GetComponent<RectTransform>().GetChild(j).gameObject.GetComponent<RectTransform>(), "Added to Questionnaire objects");
                    //        Undo.FlushUndoRecordObjects();
                    //        _pageFactory.PageList[i].GetComponent<RectTransform>().GetChild(j).gameObject.GetComponent<RectTransform>();

                    //        EditorUtility.SetDirty(_pageFactory.PageList[i].GetComponent<RectTransform>().GetChild(j).gameObject.GetComponent<RectTransform>());

                    //        AssetDatabase.Refresh();

                    //        AssetDatabase.SaveAssets();
                    //    }
                    //    //GameObject q_panel = GameObject.Find("Q_Panel");
                    //    //RectTransform qPanelRect = q_panel.GetComponent<RectTransform>();
                    //    //CenterRec(qPanelRect);
                    //}
                }

                private void GenerateQuestionnaire()
                {
                    GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                    Undo.RegisterCompleteObjectUndo(q, "Added to Questionnaire objects");
                    Undo.FlushUndoRecordObjects();

                    PrepareConfigurations();
                    InitializeQuestionnaire(q);
                    
                    EditorUtility.SetDirty(q);

                    AssetDatabase.Refresh();

                    AssetDatabase.SaveAssets();
                }

                void ReadJson(string jsonPath)
                {
                    // reads and parses .json input file
                    string JSONString = File.ReadAllText(jsonPath);
                    var N = JSON.Parse(JSONString);

                    //----------- Read metadata from .JSON file ----------//
                    string title = N["qTitle"].Value;
                    string instructions = N["qInstructions"].Value;
                    qId = N["qId"].Value; //read questionnaire ID

                    // Generates the last page
                    _pageFactory.GenerateAndDisplayFirstAndLastPage(true, instructions, title);

                    int i = 0;

                    /*
                    Continuously reads data from the .json file 
                    */
                    while (true)
                    {
                        pId = N["questions"][i]["pId"].Value; //read new page

                        if (pId != "")
                        {
                            string qType = N["questions"][i]["qType"].Value;
                            string qInstructions = N["questions"][i]["qInstructions"].Value;

                            _qData = N["questions"][i]["qData"].AsArray;
                            if (_qData == "")
                                _qData[0] = N["questions"][i]["qData"].Value;

                            _qConditions = N["questions"][i]["qConditions"].AsArray;
                            if (_qConditions == "")
                                _qConditions[0] = N["questions"][i]["qConditions"].Value;

                            _qOptions = N["questions"][i]["qOptions"].AsArray;
                            if (_qOptions == "")
                                _qOptions[0] = N["questions"][i]["qOptions"].Value;

                            _pageFactory.AddPage(qId, qType, qInstructions, _qData, _qConditions, _qOptions);
                            i++;
                        }
                        else
                        {
                            // Read data for final page from .JSON file
                            string headerFinalSlide = N["qMessage"].Value;
                            string textFinalSlide = N["qAcknowledgments"].Value;

                            // Generates the last page
                            _pageFactory.GenerateAndDisplayFirstAndLastPage(false, textFinalSlide, headerFinalSlide);

                            // Initialize (Dis-/enable GameObjects)
                            _pageFactory.InitSetup();

                            break;
                        }
                    }
                }

            }
        }
    }
}
