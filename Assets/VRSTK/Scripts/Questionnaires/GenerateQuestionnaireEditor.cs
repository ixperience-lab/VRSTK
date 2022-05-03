using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Reflection;
using System;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.Playback;
using VRQuestionnaireToolkit;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Questionnaire
        {
            public class GenerateQuestionnaireEditor : EditorWindow
            {

                public GameObject _questionnaireToolkit;
                public GameObject _lastQuestionnaireToolkit;
                //private bool[] _trackedComponents;
                //private bool[][] _trackedVariables;

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
                    if (_questionnaireToolkit != null)
                    {
                        if (_questionnaireToolkit != _lastQuestionnaireToolkit)
                        {
                            //_trackedComponents = new bool[_questionnaireToolkit.GetComponents(typeof(Component)).Length];
                            //_trackedVariables = new bool[_questionnaireToolkit.GetComponents(typeof(Component)).Length][];
                        }
                        EditorGUILayout.LabelField("Select the components and variables you want to track:");
                        //Cycle through components of the tracked object
                        //for (int i = 0; i < _questionnaireToolkit.GetComponents(typeof(Component)).Length; i++)
                        //{
                        //    Component c = _questionnaireToolkit.GetComponents(typeof(Component))[i];
                        //    if (c != null)
                        //    {
                        //        EditorStyles.label.fontStyle = FontStyle.Bold;
                        //        _trackedComponents[i] = EditorGUILayout.Toggle(c.GetType().ToString(), _trackedComponents[i]);
                        //        EditorStyles.label.fontStyle = FontStyle.Normal;
                        //        if (_questionnaireToolkit != _lastQuestionnaireToolkit)
                        //        {
                        //            _trackedVariables[i] = new bool[c.GetType().GetProperties().Length + c.GetType().GetFields().Length];
                        //        }
                        //    }

                        //    //Cycle through variables
                        //    if (_trackedComponents[i] == true)
                        //    {
                        //        EditorGUI.indentLevel++;
                        //        for (int j = 0; j < c.GetType().GetProperties().Length; j++)
                        //        {
                        //            var varToCheck = c.GetType().GetProperties()[j];
                        //            if (EventTypeChecker.IsValid(varToCheck.PropertyType))
                        //            {
                        //                _trackedVariables[i][j] = EditorGUILayout.Toggle(varToCheck.Name, _trackedVariables[i][j]);
                        //            }
                        //        }

                        //        for (int j = c.GetType().GetProperties().Length; j < c.GetType().GetFields().Length + c.GetType().GetProperties().Length; j++)
                        //        {
                        //            var varToCheck = c.GetType().GetFields()[j - c.GetType().GetProperties().Length];
                        //            if (EventTypeChecker.IsValid(varToCheck.FieldType))
                        //            {
                        //                _trackedVariables[i][j] = EditorGUILayout.Toggle(varToCheck.Name, _trackedVariables[i][j]);
                        //            }
                        //        }
                        //        EditorGUI.indentLevel--;
                        //    }
                        //}
                        if (GUILayout.Button("Generate Questionnaire"))
                        {
                            //CreateEvent();
                            GenerateQuestionnaire();
                            _questionnaireToolkit = null;
                        }
                    }
                    else
                    {
                        EditorGUILayout.LabelField("Please select a GameObject to generate a questionnaire in the inspector.");
                    }


                    _lastQuestionnaireToolkit = _questionnaireToolkit;
                }

                //private void CreateEvent()
                //{
                //    //Create Event itself
                //    Telemetry.Event newEvent = (Telemetry.Event)ScriptableObject.CreateInstance("Event");
                //    newEvent.eventName = _questionnaireToolkit.gameObject.name + _questionnaireToolkit.gameObject.GetInstanceID().ToString();

                //    int numberOfProperties = 0;
                //    int numberOfFields = 0;
                //    List<string> savedNames = new List<string>();
                //    for (int i = 0; i < _questionnaireToolkit.GetComponents(typeof(Component)).Length; i++)
                //    {
                //        Component c = _questionnaireToolkit.GetComponents(typeof(Component))[i];

                //        //Cycle through variables
                //        if (_trackedComponents[i] == true)
                //        {
                //            for (int j = 0; j < c.GetType().GetProperties().Length; j++)
                //            {
                //                if (_trackedVariables[i][j])
                //                {
                //                    savedNames.Add(string.Join("", new string[] { c.GetType().GetProperties()[j].Name, "_", c.GetType().Name }));
                //                    numberOfProperties++;
                //                }
                //            }

                //            for (int j = c.GetType().GetProperties().Length; j < c.GetType().GetFields().Length + c.GetType().GetProperties().Length; j++)
                //            {
                //                if (_trackedVariables[i][j])
                //                {
                //                    savedNames.Add(string.Join("", new string[] { c.GetType().GetFields()[j - c.GetType().GetProperties().Length].Name, "_", c.GetType().Name }));
                //                    numberOfFields++;
                //                }
                //            }
                //        }
                //    }

                //    try
                //    {
                //        TrackedObjects g = GameObject.Find("TrackedObjects").GetComponent<TrackedObjects>();
                //        Undo.RegisterCompleteObjectUndo(g, "Added to tracked objects");
                //        Undo.FlushUndoRecordObjects();
                //        g.trackedObjects.Add(_questionnaireToolkit);
                //        EditorUtility.SetDirty(g);
                //    }
                //    catch (NullReferenceException e)
                //    {
                //        Debug.LogError("The TrackedObjects GameObject was not found in the scene. Please add it to the scene from the prefabs folder. Exception message: " + e.Message);
                //    }

                //    //Attach Eventsender
                //    EventSender s = _questionnaireToolkit.AddComponent<EventSender>();
                //    s.eventBase = newEvent;
                //    s.SetTrackedVar(_trackedComponents, _trackedVariables, savedNames);
                //    AssetDatabase.CreateAsset(newEvent, "Assets/VRSTK/Events/Track" + _questionnaireToolkit.gameObject.name + _questionnaireToolkit.gameObject.GetInstanceID().ToString() + ".asset");
                //    Undo.RecordObject(newEvent, "Created Event");
                //    AssetDatabase.Refresh();
                //    s.eventBase = (Telemetry.Event)AssetDatabase.LoadAssetAtPath("Assets/VRSTK/Events/Track" + _questionnaireToolkit.gameObject.name + _questionnaireToolkit.gameObject.GetInstanceID().ToString() + ".asset", typeof(Telemetry.Event));
                //    AssetDatabase.SaveAssets();
                //}

                private void GenerateQuestionnaire()
                {
                    GenerateQuestionnaire q = _questionnaireToolkit.GetComponent<GenerateQuestionnaire>();
                    q.PrepareConfigurations();
                    q.InitializeQuestionnaire();
                }

            }
        }
    }
}
