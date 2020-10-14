using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
namespace STK
{
    ///<summary>Defines the Editor view of STKSettings.</summary>
    [CustomEditor(typeof(STKSettings))]
    public class STKSettingsEditor : Editor
    {

        public override void OnInspectorGUI()
        {
            STKSettings myTarget = (STKSettings)target;
            base.OnInspectorGUI();

            if (!myTarget.useDataReduction && !myTarget.createFileWhenFull)
            {
                myTarget.useSlidingWindow = EditorGUILayout.Toggle(new GUIContent("Use Sliding Window", "When the maximum event number is reached, an event will be removed from the beginning for each new event added."), myTarget.useSlidingWindow);
            }
            if (!myTarget.useSlidingWindow && !myTarget.createFileWhenFull)
            {
                myTarget.useDataReduction = EditorGUILayout.Toggle(new GUIContent("Use Data Reduction", "When the maximum event number is reached, every second currently stored event will be removed, reducing the precision of earlier data without removing it entirely."), myTarget.useDataReduction);
            }
            if (!myTarget.useSlidingWindow && !myTarget.useDataReduction)
            {
                myTarget.createFileWhenFull = EditorGUILayout.Toggle(new GUIContent("Save when full", "When the maximum event number is reached, a file will be created that contains all current events. Events will be cleared from memory afterwards. This will fill up your disk, so check your free space!"), myTarget.createFileWhenFull);
            }
        }
    }
}
