using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Telemetry
        {
            /// <summary>
            /// Defines the parameter of an Event. 
            /// </summary>
            [System.Serializable]
            public class EventParameter
            {
                public string name;
                public System.Type systemType;
                /// <summary>Index in the allowedTypes Array of the EventTypeChecker</summary>
                public int typeIndex;
                public bool hideFromInspector;

                public void SetTypeFromIndex()
                {
                    systemType = EventTypeChecker.allowedTypes[typeIndex];
                }
            }

            /// <summary>
            /// Event which contains custom parameters. Can be created and deployed by an EventSender component. 
            /// </summary>
            [CreateAssetMenu(menuName = "VRSTK/Event")]
            public class Event : ScriptableObject
            {
                [SerializeField]
                public List<EventParameter> parameters = new List<EventParameter>();
                /// <summary>Name of the Event. </summary>
                public string eventName;
                /// <summary>Objects are made up of a parameter name and a value</summary>
                public Hashtable objects = new Hashtable();
                /// <summary>Time the event was sent</summary>
                public float time;

                /// <summary>Defines a new Parameter</summary>
                public void AddParameter(string name, int typeIndex)
                {
                    EventParameter newParameter = new EventParameter();
                    newParameter.name = name;
                    newParameter.hideFromInspector = true;
                    newParameter.typeIndex = typeIndex;
                    parameters.Add(newParameter);
                }

                /// <summary>Sets a parameter to a certain value</summary>
                public void SetValue(string key, object value)
                {
                    //Test if Key exists and Value is the correct Datatype
                    foreach (EventParameter p in parameters)
                    {
                        if (key == p.name)
                        {
                            if (p.systemType == null)
                            {
                                p.SetTypeFromIndex();
                            }

                            if (value.GetType() == p.systemType)
                            {
                                objects.Add(key, value);
                            }
                        }
                    }

                }

            }
        }
    }
}
