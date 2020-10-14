using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.Reflection;
using System;
namespace STK
{
    ///<summary>Creates the interface for Gameobject/component tracking.</summary>
    public class STKTrackEditor : EditorWindow
    {

        public GameObject trackedObject;
        public GameObject lastTrackedObject;
        private bool[] trackedComponents;
        private bool[][] trackedVariables;

        [MenuItem("Window/VR Scientific Toolkit/Track Object")]
        public static void ShowWindow()
        {
            EditorWindow.GetWindow(typeof(STKTrackEditor));
        }

        void OnEnable()
        {
            trackedObject = Selection.activeGameObject;
            if (lastTrackedObject == null && trackedObject != null)
            {
                trackedComponents = new bool[trackedObject.GetComponents(typeof(Component)).Length];
                trackedVariables = new bool[trackedObject.GetComponents(typeof(Component)).Length][];
                //lastTrackedObject = trackedObject;
            }
        }

        private void OnInspectorUpdate()
        {
            trackedObject = Selection.activeGameObject;
            if (trackedObject != lastTrackedObject)
            {
                Repaint();
            }
        }

        private void OnGUI()
        {
            if (trackedObject != null)
            {
                if (trackedObject != lastTrackedObject)
                {
                    trackedComponents = new bool[trackedObject.GetComponents(typeof(Component)).Length];
                    trackedVariables = new bool[trackedObject.GetComponents(typeof(Component)).Length][];
                }
                EditorGUILayout.LabelField("Select the components and variables you want to track:");
                //Cycle through components of the tracked object
                for (int i = 0; i < trackedObject.GetComponents(typeof(Component)).Length; i++)
                {
                    Component c = trackedObject.GetComponents(typeof(Component))[i];
                    if (c != null)
                    {
                        EditorStyles.label.fontStyle = FontStyle.Bold;
                        trackedComponents[i] = EditorGUILayout.Toggle(c.GetType().ToString(), trackedComponents[i]);
                        EditorStyles.label.fontStyle = FontStyle.Normal;
                        if (trackedObject != lastTrackedObject)
                        {
                            trackedVariables[i] = new bool[c.GetType().GetProperties().Length + c.GetType().GetFields().Length];
                        }
                    }

                    //Cycle through variables
                    if (trackedComponents[i] == true)
                    {
                        EditorGUI.indentLevel++;
                        for (int j = 0; j < c.GetType().GetProperties().Length; j++)
                        {
                            var varToCheck = c.GetType().GetProperties()[j];
                            if (STKEventTypeChecker.IsValid(varToCheck.PropertyType))
                            {
                                trackedVariables[i][j] = EditorGUILayout.Toggle(varToCheck.Name, trackedVariables[i][j]);
                            }
                        }

                        for (int j = c.GetType().GetProperties().Length; j < c.GetType().GetFields().Length + c.GetType().GetProperties().Length; j++)
                        {
                            var varToCheck = c.GetType().GetFields()[j - c.GetType().GetProperties().Length];
                            if (STKEventTypeChecker.IsValid(varToCheck.FieldType))
                            {
                                trackedVariables[i][j] = EditorGUILayout.Toggle(varToCheck.Name, trackedVariables[i][j]);
                            }
                        }
                        EditorGUI.indentLevel--;
                    }
                }
                if (GUILayout.Button("Create Tracker"))
                {
                    CreateEvent();
                    trackedObject = null;
                }
            }
            else
            {
                EditorGUILayout.LabelField("Please select a GameObject to track in the inspector.");
            }


            lastTrackedObject = trackedObject;
        }

        private void CreateEvent()
        {
            //Create Event itself
            STKEvent newEvent = (STKEvent)ScriptableObject.CreateInstance("STKEvent");
            newEvent.eventName = trackedObject.gameObject.name + trackedObject.gameObject.GetInstanceID().ToString();

            int numberOfProperties = 0;
            int numberOfFields = 0;
            List<string> savedNames = new List<string>();
            for (int i = 0; i < trackedObject.GetComponents(typeof(Component)).Length; i++)
            {
                Component c = trackedObject.GetComponents(typeof(Component))[i];

                //Cycle through variables
                if (trackedComponents[i] == true)
                {
                    for (int j = 0; j < c.GetType().GetProperties().Length; j++)
                    {
                        if (trackedVariables[i][j])
                        {
                            savedNames.Add(string.Join("", new string[] { c.GetType().GetProperties()[j].Name, "_", c.GetType().Name }));
                            numberOfProperties++;
                        }
                    }

                    for (int j = c.GetType().GetProperties().Length; j < c.GetType().GetFields().Length + c.GetType().GetProperties().Length; j++)
                    {
                        if (trackedVariables[i][j])
                        {
                            savedNames.Add(string.Join("", new string[] { c.GetType().GetFields()[j - c.GetType().GetProperties().Length].Name, "_", c.GetType().Name }));
                            numberOfFields++;
                        }
                    }
                }
            }

            try
            {
                STKTrackedObjects g = GameObject.Find("STKTrackedObjects").GetComponent<STKTrackedObjects>();
                Undo.RegisterCompleteObjectUndo(g, "Added to tracked objects");
                Undo.FlushUndoRecordObjects();
                g.trackedObjects.Add(trackedObject);
                EditorUtility.SetDirty(g);
            }
            catch (NullReferenceException e)
            {
                Debug.LogError("The STKTrackedObjects GameObject was not found in the scene. Please add it to the scene from the prefabs folder.");
            }

            //Attach Eventsender
            STKEventSender s = trackedObject.AddComponent<STKEventSender>();
            s.eventBase = newEvent;
            s.SetTrackedVar(trackedComponents, trackedVariables, savedNames);
            AssetDatabase.CreateAsset(newEvent, "Assets/VRScientificToolkit/Events/Track" + trackedObject.gameObject.name + trackedObject.gameObject.GetInstanceID().ToString() + ".asset");
            Undo.RecordObject(newEvent, "Created Event");
            AssetDatabase.Refresh();
            s.eventBase = (STKEvent)AssetDatabase.LoadAssetAtPath("Assets/VRScientificToolkit/Events/Track" + trackedObject.gameObject.name + trackedObject.gameObject.GetInstanceID().ToString() + ".asset", typeof(STKEvent));
            AssetDatabase.SaveAssets();
        }

    }
}
