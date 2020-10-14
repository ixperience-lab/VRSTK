using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
namespace STK
{
    ///<summary>Creates Interface on stages for creation of properties</summary>
    [CustomEditor(typeof(STKTestStage))]
    public class STKTestStageEditor : Editor
    {

        public string propertyName;
        public string buttonName;
        public bool startProperty = true;

        public override void OnInspectorGUI()
        {
            if (!EditorApplication.isPlaying)
            {
                STKTestStage myTarget = (STKTestStage)target;
                EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());

                EditorGUILayout.LabelField("Add a new Property:");
                propertyName = EditorGUILayout.TextField("Name of new Property: ", propertyName);

                if (propertyName != null && propertyName != "")
                {
                    startProperty = EditorGUILayout.Toggle(new GUIContent("Start Property", "A start property is put in before the stage is started. Otherwise it can be put in while the stage is running."), startProperty);
                    if (GUILayout.Button("Add Input Property"))
                    {
                        myTarget.AddInputProperty(propertyName, startProperty);
                        Debug.Log("Edit " + startProperty);
                        propertyName = "";
                    }
                    if (GUILayout.Button("Add Toggle Property"))
                    {
                        myTarget.AddToggleProperty(propertyName, startProperty);
                        propertyName = "";
                    }
                }
                else
                {
                    GUILayout.Label("Please choose a name before adding a new property.");
                }
                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Add a new Button:");
                buttonName = EditorGUILayout.TextField("Name of new Button: ", buttonName);

                if (buttonName != null && buttonName != "")
                {
                    if (GUILayout.Button("Add Button"))
                    {
                        myTarget.AddButton(buttonName);
                        buttonName = "";
                    }
                }
                else
                {
                    GUILayout.Label("Please choose a name before adding a new button.");
                }
                EditorGUILayout.Space();

                base.OnInspectorGUI();
            }
        }

    }
}
