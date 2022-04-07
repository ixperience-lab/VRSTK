using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SimpleJSON;
using VRSTK.Scripts.Telemetry;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Playback
        {
            ///<summary>Sceneplayback recreates Scene states from JSON data</summary>
            public static class ScenePlayback
            {

                public static bool playbackMode;
                private static List<GameObject> trackedObjects;
                private static JSONNode parsedJson;
                private static int stage;

                public static void StartPlayback(string json, int stageToPlay) //Starts when a JSON File is loaded
                {
                    System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
                    playbackMode = true;
                    GameObject.Find("TrackedObjects").GetComponent<TrackedObjects>().CheckForNullReferences();
                    trackedObjects = GameObject.Find("TrackedObjects").GetComponent<TrackedObjects>().trackedObjects;
                    parsedJson = JSON.Parse(json);
                    stage = stageToPlay;
                }

                //Called from PlaybackEditor
                public static void DeactivateAllComponents() 
                {
                    GameObject[] allGameObjects = Object.FindObjectsOfType<GameObject>();

                    foreach (GameObject go in allGameObjects)
                    {
                        MonoBehaviour[] components = go.GetComponents<MonoBehaviour>();
                        foreach (MonoBehaviour c in components)
                        {
                            c.enabled = false;
                        }
                        Rigidbody[] rigidbodies = go.GetComponents<Rigidbody>();
                        foreach (Rigidbody r in rigidbodies)
                        {
                            r.isKinematic = true;
                        }
                    }
                }

                public static void GoToPoint(float t)
                {
                    foreach (GameObject g in trackedObjects) // Goes through events with Eventsenders and finds their respective events in the JSON file
                    {
                        if (g != null && g.GetComponent<EventSender>() != null && g.GetComponent<EventSender>().eventBase != null)
                        {
                            EventSender[] senders = g.GetComponents<EventSender>();
                            g.SetActive(false);
                            foreach (EventSender s in senders)
                            {
                                Telemetry.Event eventBase = s.eventBase;
                                JSONNode currentEvent = parsedJson;
                                JSONNode events = parsedJson[("Stage" + stage.ToString())][eventBase.eventName];
                                if (events != null)
                                {
                                    if (g.activeSelf == false)
                                    {
                                        g.SetActive(true);
                                        MonoBehaviour[] components = g.GetComponents<MonoBehaviour>();
                                        foreach (MonoBehaviour c in components)
                                        {
                                            c.enabled = false;
                                        }
                                        Rigidbody[] rigidbodies = g.GetComponents<Rigidbody>();
                                        foreach (Rigidbody r in rigidbodies)
                                        {
                                            r.isKinematic = true;
                                        }
                                    }
                                    for (int i = 0; i < events.Count; i++)
                                    {
                                        if (events[i]["time"] >= t) //Finds event closest in time to point that will be restored
                                        {
                                            currentEvent = events[i];
                                            i = events.Count;
                                        }
                                    }
                                    foreach (EventParameter param in eventBase.parameters)
                                    {
                                        Component component = s.GetComponentFromParameter(param.name);
                                        string name = s.GetVariableNameFromEventVariable(param.name);
                                        if (name != null && name != "")
                                        {
                                            SetVariable(currentEvent[param.name], name, component, g);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                public static float GoToNextPoint(float currentTime)
                {
                    foreach (GameObject g in trackedObjects) // Goes through events with Eventsenders and finds their respective events in the JSON file
                    {
                        if (g != null && g.GetComponent<EventSender>() != null && g.GetComponent<EventSender>().eventBase != null)
                        {
                            if (g.activeSelf == false)
                            {
                                g.SetActive(true);
                                MonoBehaviour[] components = g.GetComponents<MonoBehaviour>();
                                foreach (MonoBehaviour c in components)
                                {
                                    c.enabled = false;
                                }
                                Rigidbody[] rigidbodies = g.GetComponents<Rigidbody>();
                                foreach (Rigidbody r in rigidbodies)
                                {
                                    r.isKinematic = true;
                                }
                            }
                            EventSender[] senders = g.GetComponents<EventSender>();
                            g.SetActive(false);
                            foreach (EventSender s in senders)
                            {
                                Telemetry.Event eventBase = s.eventBase;
                                JSONNode currentEvent = parsedJson;
                                JSONNode events = parsedJson[("Stage" + stage.ToString())][eventBase.eventName];
                                if (events != null)
                                {
                                    string newFrameTime = "0";
                                    for (int i = 0; i < events.Count; i++)
                                    {
                                        if (float.Parse(events[i]["time"]) > currentTime)
                                        {
                                            currentEvent = events[i];
                                            newFrameTime = events[i]["time"];
                                            break;
                                        }
                                    }
                                    foreach (EventParameter param in eventBase.parameters)
                                    {
                                        Component component = s.GetComponentFromParameter(param.name);
                                        string name = s.GetVariableNameFromEventVariable(param.name);
                                        if (name != null && name != "")
                                        {
                                            SetVariable(currentEvent[param.name], name, component, g);
                                        }
                                    }
                                    return float.Parse(newFrameTime);
                                }
                            }
                        }
                    }
                    return 0.0f;
                }

                public static float GoToPreviousPoint(float currentTime)
                {
                    if (currentTime < 0)
                    {
                        return 0.0f;
                    }
                    foreach (GameObject g in trackedObjects) // Goes through events with Eventsenders and finds their respective events in the JSON file
                    {
                        if (g != null && g.GetComponent<EventSender>() != null && g.GetComponent<EventSender>().eventBase != null)
                        {
                            if (g.activeSelf == false)
                            {
                                g.SetActive(true);
                                MonoBehaviour[] components = g.GetComponents<MonoBehaviour>();
                                foreach (MonoBehaviour c in components)
                                {
                                    c.enabled = false;
                                }
                                Rigidbody[] rigidbodies = g.GetComponents<Rigidbody>();
                                foreach (Rigidbody r in rigidbodies)
                                {
                                    r.isKinematic = true;
                                }
                            }
                            EventSender[] senders = g.GetComponents<EventSender>();
                            g.SetActive(false);
                            foreach (EventSender s in senders)
                            {
                                Telemetry.Event eventBase = s.eventBase;
                                JSONNode currentEvent = parsedJson;
                                JSONNode events = parsedJson[("Stage" + stage.ToString())][eventBase.eventName];
                                if (events != null)
                                {
                                    string newFrameTime = "0";
                                    for (int i = 0; i < events.Count; i++)
                                    {
                                        if (float.Parse(events[i]["time"]) >= currentTime)
                                        {
                                            if (i == 0)
                                            {
                                                currentEvent = events[i];
                                                newFrameTime = "0";
                                            }
                                            else
                                            {
                                                currentEvent = events[i - 1];
                                                newFrameTime = events[i - 1]["time"];
                                            }
                                            break;

                                        }
                                    }

                                    foreach (EventParameter param in eventBase.parameters)
                                    {
                                        Component component = s.GetComponentFromParameter(param.name);
                                        string name = s.GetVariableNameFromEventVariable(param.name);
                                        if (name != null && name != "")
                                        {
                                            SetVariable(currentEvent[param.name], name, component, g);
                                        }
                                    }
                                    return float.Parse(newFrameTime);
                                }
                            }
                        }
                    }
                    return 0.0f;
                }

                public static float GetLastTimestampOfCurrentStage()
                {
                    foreach (GameObject g in trackedObjects) // Goes through events with Eventsenders and finds their respective events in the JSON file
                    {
                        if (g != null && g.GetComponent<EventSender>() != null && g.GetComponent<EventSender>().eventBase != null)
                        {
                            EventSender[] senders = g.GetComponents<EventSender>();
                            g.SetActive(false);
                            foreach (EventSender s in senders)
                            {
                                Telemetry.Event eventBase = s.eventBase;
                                JSONNode currentEvent = parsedJson;
                                JSONNode events = parsedJson[("Stage" + stage.ToString())][eventBase.eventName];
                                if (events != null)
                                {
                                    string timestamp = events[events.Count - 1]["time"];
                                    if (timestamp != null)
                                    {
                                        return float.Parse(timestamp);
                                    }
                                }
                            }
                        }
                    }
                    return 0;
                }

                public static void SetVariable(JSONNode node, string name, Component c, GameObject g) //Sets ariable of a gameobject to a value from the JSON file
                {
                    if (node.IsArray) //Arrays are converted into vectors
                    {
                        switch (node.Count)
                        {
                            case 2:
                                Vector2 v2 = new Vector2(node[0], node[1]);
                                if (c.GetType().GetField(name) != null)
                                {
                                    c.GetType().GetField(name).SetValue(c, v2);
                                }
                                else
                                {
                                    c.GetType().GetProperty(name).SetValue(c, v2);
                                }
                                break;
                            case 3:
                                Vector3 v3 = new Vector3(node[0], node[1], node[2]);
                                if (c.GetType().GetField(name) != null)
                                {
                                    c.GetType().GetField(name).SetValue(c, v3);
                                }
                                else
                                {
                                    c.GetType().GetProperty(name).SetValue(c, v3);
                                }
                                break;
                            case 4:
                                if (c.GetType().GetField(name) != null)
                                {
                                    if (c.GetType().GetField(name).GetValue(c).GetType() == typeof(Vector4))
                                    {
                                        Vector4 v4 = new Vector4(node[0], node[1], node[2], node[3]);
                                        c.GetType().GetField(name).SetValue(c, v4);
                                    }
                                    else if (c.GetType().GetField(name).GetValue(c).GetType() == typeof(Quaternion))
                                    {
                                        Quaternion q = new Quaternion(node[0], node[1], node[2], node[3]);
                                        c.GetType().GetField(name).SetValue(c, q);
                                    }
                                }
                                else
                                {
                                    if (c.GetType().GetProperty(name).GetValue(c).GetType() == typeof(Vector4))
                                    {
                                        Vector4 v4 = new Vector4(node[0], node[1], node[2], node[3]);
                                        c.GetType().GetProperty(name).SetValue(c, v4);
                                    }
                                    else if (c.GetType().GetProperty(name).GetValue(c).GetType() == typeof(Quaternion))
                                    {
                                        Quaternion q = new Quaternion(node[0], node[1], node[2], node[3]);
                                        c.GetType().GetProperty(name).SetValue(c, q);
                                    }
                                }
                                break;
                        }
                    }
                    else if (node.IsBoolean)
                    {
                        bool b = node;
                        if (c.GetType().GetField(name) != null)
                        {
                            c.GetType().GetField(name).SetValue(c, b);
                        }
                        else
                        {
                            c.GetType().GetProperty(name).SetValue(c, b);
                        }
                    }
                    else if (node.IsNumber)
                    {
                        float f = node;
                        if (c.GetType().GetField(name) != null)
                        {
                            if (c.GetType().GetField(name).GetValue(c).GetType() == typeof(int))
                            {
                                int i = (int)f;
                                c.GetType().GetField(name).SetValue(c, i);
                            }
                            else
                            {
                                c.GetType().GetField(name).SetValue(c, f);
                            }
                        }
                        else
                        {
                            if (c.GetType().GetProperty(name).GetValue(c).GetType() == typeof(int))
                            {
                                int i = (int)f;
                                c.GetType().GetProperty(name).SetValue(c, i);
                            }
                            else
                            {
                                c.GetType().GetProperty(name).SetValue(c, f);
                            }
                        }
                    }
                    else if (node.IsString)
                    {
                        string s = node;
                        if (c.GetType().GetField(name) != null)
                        {
                            c.GetType().GetField(name).SetValue(c, s);
                        }
                        else
                        {
                            // hewl ToDo: Converting "True" to boolean true
                            if (c.GetType() == typeof(bool) || c.GetType() == typeof(System.Boolean))
                            {
                                bool b = node;
                                c.GetType().GetProperty(name).SetValue(c, b);
                            }
                            else
                                c.GetType().GetProperty(name).SetValue(c, s);
                        }
                    }
                }

                public static void StopPlayback()
                {
                    playbackMode = false;
                }
            }
        }
    }
}
