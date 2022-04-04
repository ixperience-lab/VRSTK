using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace EditorControl
        {
            ///<summary>Creates button to spawn new stage.</summary>
            [CustomEditor(typeof(TestController))]
            public class TestControllerEditor : UnityEditor.Editor
            {

                public override void OnInspectorGUI()
                {
                    DrawDefaultInspector();

                    if (!EditorApplication.isPlaying)
                    {
                        TestController myTarget = (TestController)target;
                        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                        GUILayout.Space(20f);
                        if (GUILayout.Button(new GUIContent("Add Stage", "Add a stage to your experiment. Different stages can have different attributes.")))
                        {
                            myTarget.AddStage();
                        }
                        GUILayout.Space(20f);
                    }

                }
            }
        }
    }
}
