using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Reflection;
using UnityEditor;

namespace STK
{
    ///<summary>Contains an array of string arrays that can be serialized.</summary>
    [System.Serializable]
    public class DoubleStringArray
    {
        [System.Serializable]
        public class StringArray
        {
            public string[] array;
        }
        public StringArray[] array;
    }

    ///<summary>Sends events out from a GameObject.</summary>
    public class STKEventSender : MonoBehaviour
    {
        ///<summary>Base event that will be cloned for deployment.</summary>
        public STKEvent eventBase;
        ///<summary>Cloned version of the base event that will be sent to the event receiver..</summary>
        [HideInInspector]
        public STKEvent eventToSend;
        public Component[] trackedComponents;
        ///<summary>Name to find in component</summary>
        [SerializeField]
        public DoubleStringArray trackedVariableNames;
        ///<summary>Name + identifier in event</summary>
        [SerializeField]
        public DoubleStringArray eventVariableNames; 
        public bool timedInterval;
        public float interval = 1;
        private float timeToSend;

        private void Start()
        {
            timeToSend = interval;
            eventToSend = Instantiate(eventBase);
        }

        private void Update()
        {
            if (timedInterval && STKTestStage.GetStarted())
            {
                timeToSend -= Time.deltaTime;
                if (timeToSend < 0)
                {
                    Deploy();
                    timeToSend = interval;
                }
            }
        }

        public void SetEventValue(string name, object o)
        {
            eventToSend.SetValue(name, o);
        }

        ///<summary>Sends event to event receiver.</summary>
        [ContextMenu("Deploy")]
        public void Deploy()
        {
            if (trackedComponents != null) //Get Values if this is a tracker
            {
                for (int i = 0; i < trackedComponents.Length; i++)
                {
                    for (int j = 0; j < trackedVariableNames.array[i].array.Length; j++)
                    {
                        if (trackedComponents[i].GetType().GetProperty(trackedVariableNames.array[i].array[j]) != null)
                        {
                            eventToSend.SetValue(eventVariableNames.array[i].array[j], trackedComponents[i].GetType().GetProperty(trackedVariableNames.array[i].array[j]).GetValue(trackedComponents[i]));
                        }
                        else if (trackedComponents[i].GetType().GetField(trackedVariableNames.array[i].array[j]) != null)
                        {
                            eventToSend.SetValue(eventVariableNames.array[i].array[j], trackedComponents[i].GetType().GetField(trackedVariableNames.array[i].array[j]).GetValue(trackedComponents[i]));
                        }
                    }

                }
            }

            eventToSend.time = STKTestStage.GetTime();
            STKEventReceiver.ReceiveEvent(eventToSend);
            eventToSend = Instantiate(eventBase);
        }

        ///<summary>Sets references to the tracked variables of this Gameobject.</summary>
        public void SetTrackedVar(bool[] comps, bool[][] vars, List<string> eventVarNames)
        {
            int numberoftrackedComps = 0;

            foreach (bool c in comps)
            {
                if (c)
                {
                    numberoftrackedComps++;
                }
            }

            trackedComponents = new Component[numberoftrackedComps];
            trackedVariableNames = new DoubleStringArray();
            trackedVariableNames.array = new DoubleStringArray.StringArray[numberoftrackedComps];

            eventVariableNames = new DoubleStringArray();
            eventVariableNames.array = new DoubleStringArray.StringArray[numberoftrackedComps];

            for (int i = 0; i < numberoftrackedComps; i++)
            {
                trackedVariableNames.array[i] = new DoubleStringArray.StringArray();
                eventVariableNames.array[i] = new DoubleStringArray.StringArray();
            }

            int eventVariableIndex = 0;
            int trackedCompsIndex = 0;
            for (int i = 0; i < comps.Length; i++)
            {
                if (comps[i])
                {
                    trackedComponents[trackedCompsIndex] = GetComponents(typeof(Component))[i];

                    int numberofTrackedVars = 0;
                    foreach (bool b in vars[i])
                    {
                        if (b)
                        {
                            numberofTrackedVars++;
                        }
                    }
                    trackedVariableNames.array[trackedCompsIndex].array = new string[numberofTrackedVars];
                    eventVariableNames.array[trackedCompsIndex].array = new string[numberofTrackedVars];

                    int varNameIndex = 0;

                    for (int j = 0; j < vars[i].Length; j++)
                    {
                        if (vars[i][j])
                        {
                            if (j >= trackedComponents[trackedCompsIndex].GetType().GetProperties().Length)
                            {
                                trackedVariableNames.array[trackedCompsIndex].array[varNameIndex] = trackedComponents[trackedCompsIndex].GetType().GetFields()[j - trackedComponents[trackedCompsIndex].GetType().GetProperties().Length].Name;
                                eventVariableNames.array[trackedCompsIndex].array[varNameIndex] = eventVarNames[eventVariableIndex];
                                eventBase.AddParameter(eventVarNames[eventVariableIndex], STKEventTypeChecker.getIndex(trackedComponents[trackedCompsIndex].GetType().GetField(trackedVariableNames.array[trackedCompsIndex].array[varNameIndex]).GetValue(trackedComponents[trackedCompsIndex]).GetType()));
                            }
                            else
                            {
                                trackedVariableNames.array[trackedCompsIndex].array[varNameIndex] = trackedComponents[trackedCompsIndex].GetType().GetProperties()[j].Name;
                                eventVariableNames.array[trackedCompsIndex].array[varNameIndex] = eventVarNames[eventVariableIndex];
                                eventBase.AddParameter(eventVarNames[eventVariableIndex], STKEventTypeChecker.getIndex(trackedComponents[trackedCompsIndex].GetType().GetProperty(trackedVariableNames.array[trackedCompsIndex].array[varNameIndex]).GetValue(trackedComponents[trackedCompsIndex]).GetType()));
                            }
                            varNameIndex++;
                            eventVariableIndex++;
                        }
                    }

                    trackedCompsIndex++;
                }
            }
        }

        ///<summary>Returns the component a parameter refers to.</summary>
        public Component GetComponentFromParameter(string parameterName)
        {
            for (int i = 0; i < trackedComponents.Length; i++)
            {
                for (int j = 0; j < eventVariableNames.array[i].array.Length; j++)
                {
                    if (eventVariableNames.array[i].array[j] == parameterName)
                    {
                        return trackedComponents[i];
                    }
                }
            }
            return GetComponent<STKEventSender>();
        }

        ///<summary>Returns the regular variable name from an event property.</summary>
        public string GetVariableNameFromEventVariable(string eventVName)
        {
            for (int i = 0; i < trackedComponents.Length; i++)
            {
                for (int j = 0; j < eventVariableNames.array[i].array.Length; j++)
                {
                    if (eventVariableNames.array[i].array[j] == eventVName)
                    {
                        return trackedVariableNames.array[i].array[j];
                    }
                }
            }
            return "";
        }
    }
}


