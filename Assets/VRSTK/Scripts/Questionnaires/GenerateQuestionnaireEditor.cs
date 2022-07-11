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

                //private List<string> _jsonInputFiles;

                public int _indexOfAQuestionnaireToSetActive = 0;

                //private PageFactory _pageFactory;
                //private ExportToCSV _exportToCsvScript;
                
                //private VRSTK.Scripts.JSONArray _qData;
                //private VRSTK.Scripts.JSONArray _qConditions;
                //private VRSTK.Scripts.JSONArray _qOptions;

                //private GameObject currentQuestionnaire;
                //private int numberQuestionnaires;
                //private string qId;
                //private string pId;

                [MenuItem("Window/VRSTK/Generate Questionnaire")]
                public static void ShowWindow()
                {
                    EditorWindow.GetWindow(typeof(GenerateQuestionnaireEditor));
                }

                void OnEnable()
                {
                    _questionnaireToolkit = Selection.activeGameObject;
                    if (_lastQuestionnaireToolkit == null && _questionnaireToolkit != null && _questionnaireToolkit.name.Contains("VRQuestionnaireToolkit"))
                    {
                        GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                        _jsonInputPath_1 = q.JsonInputPath_1;
                        _jsonInputPath_2 = q.JsonInputPath_2;
                        _jsonInputPath_3 = q.JsonInputPath_3;
                        _jsonInputPath_4 = q.JsonInputPath_4;
                        _jsonInputPath_5 = q.JsonInputPath_5;
                        _jsonInputPath_6 = q.JsonInputPath_6;
                        _jsonInputPath_7 = q.JsonInputPath_7;
                        _jsonInputPath_8 = q.JsonInputPath_8;
                        _jsonInputPath_9 = q.JsonInputPath_9;
                        _jsonInputPath_10 = q.JsonInputPath_10;
                        _indexOfAQuestionnaireToSetActive = q._indexOfAQuestionnaireToSetActive;
                    }
                }

                private void OnInspectorUpdate()
                {
                    _questionnaireToolkit = Selection.activeGameObject;
                    if (_questionnaireToolkit != _lastQuestionnaireToolkit && _questionnaireToolkit != null && _questionnaireToolkit.name.Contains("VRQuestionnaireToolkit"))
                    {
                        GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                        _jsonInputPath_1 = q.JsonInputPath_1;
                        _jsonInputPath_2 = q.JsonInputPath_2;
                        _jsonInputPath_3 = q.JsonInputPath_3;
                        _jsonInputPath_4 = q.JsonInputPath_4;
                        _jsonInputPath_5 = q.JsonInputPath_5;
                        _jsonInputPath_6 = q.JsonInputPath_6;
                        _jsonInputPath_7 = q.JsonInputPath_7;
                        _jsonInputPath_8 = q.JsonInputPath_8;
                        _jsonInputPath_9 = q.JsonInputPath_9;
                        _jsonInputPath_10 = q.JsonInputPath_10;
                        _indexOfAQuestionnaireToSetActive = q._indexOfAQuestionnaireToSetActive;
                        Repaint();
                    }
                }

                private void OnGUI()
                {
                    if (_questionnaireToolkit != null && _questionnaireToolkit.name.Contains("VRQuestionnaireToolkit"))
                    {
                        _widht = position.width;
                        _height = position.height;

                        _scrollPosition = GUILayout.BeginScrollView(_scrollPosition, true, true, GUILayout.Width(_widht), GUILayout.Height(_height));

                        
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

                void DeleteQuestionnaire(GenerateQuestionnaire q)
                {
                    Undo.RegisterCompleteObjectUndo(q, "Added to Questionnaire objects");
                    Undo.FlushUndoRecordObjects();

                    for (int i = 0; i < q.Questionnaires.Count; i++)
                    {
                        _questionnaireToolkit.transform.Find(q.Questionnaires[i].name).gameObject.SetActive(false);
                        DestroyImmediate(_questionnaireToolkit.transform.Find(q.Questionnaires[i].name).gameObject);
                    }
                    q.Questionnaires.Clear();

                    EditorUtility.SetDirty(q);

                    AssetDatabase.Refresh();

                    AssetDatabase.SaveAssets();
                }

                private void GenerateQuestionnaire()
                {
                    GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                    Undo.RegisterCompleteObjectUndo(q, "Added to Questionnaire objects");
                    Undo.FlushUndoRecordObjects();
                     
                    q.JsonInputPath_1 = _jsonInputPath_1;
                    q.JsonInputPath_2 = _jsonInputPath_2;
                    q.JsonInputPath_3 = _jsonInputPath_3;
                    q.JsonInputPath_4 = _jsonInputPath_4;
                    q.JsonInputPath_5 = _jsonInputPath_5;
                    q.JsonInputPath_6 = _jsonInputPath_6;
                    q.JsonInputPath_7 = _jsonInputPath_7;
                    q.JsonInputPath_8 = _jsonInputPath_8;
                    q.JsonInputPath_9 = _jsonInputPath_9;
                    q.JsonInputPath_10 = _jsonInputPath_10;
                    q._indexOfAQuestionnaireToSetActive = _indexOfAQuestionnaireToSetActive;

                    q.PrepareConfigurations();//PrepareConfigurations();
                    q.InitializeQuestionnaire();//InitializeQuestionnaire(q);
                    
                    EditorUtility.SetDirty(q);

                    AssetDatabase.Refresh();

                    AssetDatabase.SaveAssets();
                }

            }
        }
    }
}
