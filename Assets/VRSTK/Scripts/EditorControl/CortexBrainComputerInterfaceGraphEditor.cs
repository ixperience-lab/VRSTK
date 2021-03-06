using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using VRSTK.Scripts.VRIntegration;

namespace VRSTK
{
    namespace Scripts
    {
        namespace EditorControl
        {
            ///<summary>Creates button to spawn new stage.</summary>
            [CustomEditor(typeof(CortexBrainComputerInterfacePrototype))]
            public class CortexBrainComputerInterfaceGraphEditor : UnityEditor.Editor
            {
                public override void OnInspectorGUI()
                {
                    base.OnInspectorGUI();

                    //EditorGraph graph = new EditorGraph(0, -1, 10, 1, "Just a sin wave", 100);
                    //graph.AddFunction(x => Mathf.Sin(x));
                    //graph.Draw();

                    //DrawDefaultInspector();

                    //if (!EditorApplication.isPlaying)
                    //{
                    //    TestController myTarget = (TestController)target;
                    //    EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
                    //    GUILayout.Space(20f);
                    //    if (GUILayout.Button(new GUIContent("Add Stage", "Add a stage to your experiment. Different stages can have different attributes.")))
                    //    {
                    //        myTarget.AddStage();
                    //    }
                    //    GUILayout.Space(20f);
                    //}

                }
            }
        }
    }
}
