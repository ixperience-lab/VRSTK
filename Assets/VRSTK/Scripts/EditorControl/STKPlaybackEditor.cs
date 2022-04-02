using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
using VRSTK.Scripts.Playback;

namespace VRSTK
{
    namespace Scripts
    {
        namespace EditorControl
        {
            ///<summary>Creates controls for the scene playback function.</summary>
            public class STKPlaybackEditor : EditorWindow
            {

                [MenuItem("Window/VR Scientific Toolkit/JSON Playback")]
                public static void ShowWindow()
                {
                    EditorWindow.GetWindow(typeof(STKPlaybackEditor));
                }

                int currentStage;
                int lastStage;
                float currentTime;
                float lastTime;
                float lastTimestampOfStage;
                string filePath;
                bool started;
                bool playing = false;
                bool steppingForward = false;
                bool steppingBackward = false;
                bool reInitialize = false;
                float lastSystemTime;
                float playbackSpeed = 1.0f;

                private void OnInspectorUpdate()
                {
                    float t = Time.realtimeSinceStartup;
                    if (currentTime < lastTimestampOfStage)
                    {
                        if (playing)
                        {
                            currentTime += (t - lastSystemTime) * playbackSpeed;
                            STKScenePlayback.GoToPoint(currentTime);
                            Repaint();
                        }
                        else if (steppingForward)
                        {
                            steppingForward = false;
                            currentTime = STKScenePlayback.GoToNextPoint(currentTime);
                            Repaint();
                        }
                    }
                    else
                    {
                        currentTime = lastTimestampOfStage;
                        playing = false;
                        Repaint();
                    }

                    if (currentTime >= 0)
                    {
                        if (steppingBackward)
                        {
                            steppingBackward = false;
                            currentTime = STKScenePlayback.GoToPreviousPoint(currentTime);
                            Repaint();
                        }
                    }
                    else
                    {
                        currentTime = 0;
                    }

                    if (reInitialize)
                    {
                        InitializeValues();
                        Repaint();
                        reInitialize = false;
                    }

                    lastSystemTime = t;
                }


                private void InitializeValues()
                {
                    currentStage = 0;
                    lastStage = 0;
                    lastTime = 0;
                    currentTime = 0;
                    playing = false;
                    lastSystemTime = Time.realtimeSinceStartup;
                    playbackSpeed = 1.0f;
                    if (filePath != null)
                    {
                        StreamReader reader = new StreamReader(filePath);
                        string s = reader.ReadToEnd();
                        STKScenePlayback.StartPlayback(s, currentStage);
                        STKScenePlayback.GoToPoint(currentTime);
                        lastTimestampOfStage = STKScenePlayback.GetLastTimestampOfCurrentStage();
                    }
                }


                private void OnGUI()
                {
                    if (!EditorApplication.isPlaying)
                    {
                        EditorGUILayout.LabelField("Please enter play mode to use the Playback tool.");
                        lastTimestampOfStage = 0;
                        currentStage = 0;
                        lastStage = -1;
                        currentTime = 0;
                        lastTime = -1;
                        filePath = null;
                        started = false;
                        playing = false;
                    }
                    else
                    {
                        EditorGUILayout.LabelField("Playback");
                        EditorGUILayout.Space();


                        if (GUILayout.Button("Select JSON File"))
                        {
                            System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
                            filePath = EditorUtility.OpenFilePanel("Select JSON File", filePath, "json");
                            InitializeValues();
                        }
                        EditorGUILayout.Space();

                        if (!started)
                        {
                            STKScenePlayback.DeactivateAllComponents();
                            started = true;
                        }

                        if (filePath != null && filePath != "")
                        {
                            currentStage = EditorGUILayout.IntField("Stage (Start at 0):", currentStage);
                            currentTime = EditorGUILayout.FloatField("Time to Restore:", currentTime);
                            playbackSpeed = EditorGUILayout.FloatField("Playback Speed: ", playbackSpeed);

                            EditorGUILayout.LabelField("Last Recorded Time in Stage: " + lastTimestampOfStage);

                            if (currentStage != lastStage)
                            {
                                lastStage = currentStage;
                                StreamReader reader = new StreamReader(filePath);
                                string s = reader.ReadToEnd();
                                STKScenePlayback.StartPlayback(s, currentStage);
                                STKScenePlayback.GoToPoint(currentTime);

                                lastTimestampOfStage = STKScenePlayback.GetLastTimestampOfCurrentStage();
                            }

                            else if (currentTime != lastTime)
                            {
                                STKScenePlayback.GoToPoint(currentTime);
                            }

                            EditorGUILayout.Space();
                            if (!playing)
                            {
                                if (GUILayout.Button("Play"))
                                {
                                    playing = true;
                                }

                                GUILayout.BeginHorizontal();
                                if (GUILayout.Button("Step Backwards"))
                                {
                                    steppingBackward = true;
                                }

                                if (GUILayout.Button("Step Forwards"))
                                {
                                    steppingForward = true;
                                }
                                GUILayout.EndHorizontal();
                            }
                            else
                            {
                                if (GUILayout.Button("Pause"))
                                {
                                    playing = false;
                                }
                            }

                            EditorGUILayout.Space();
                            if (GUILayout.Button("Reset Values"))
                            {
                                InitializeValues();
                            }


                            lastStage = currentStage;
                            lastTime = currentTime;
                        }
                    }
                }
            }
        }
    }
}
