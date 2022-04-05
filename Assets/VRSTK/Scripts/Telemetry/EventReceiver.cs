using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Telemetry
        {
            /// <summary>Collects all Events to later send them to JSONParser.</summary>
            public static class EventReceiver
            {
                /// <summary>Contains all events that were collected.</summary>
                public static Hashtable savedEvents = new Hashtable();
                private static Settings settings = Resources.Load<Settings>("Settings");

                public static void ReceiveEvent(Event e)
                {
                    if (savedEvents.ContainsKey(e.eventName))
                    {
                        List<Event> eventsList = (List<Event>)savedEvents[e.eventName];
                        eventsList.Add(e);
                        savedEvents[e.eventName] = eventsList;
                        if (settings.useSlidingWindow && eventsList.Count > settings.EventMaximum) //Reduces Data volume when too many Events were received
                        {
                            eventsList.RemoveAt(0); //Removes first Element (Sliding window)
                        }
                        else if (settings.useDataReduction && eventsList.Count > settings.EventMaximum)
                        {
                            eventsList = ReduceListData(eventsList);
                        }
                        else if (settings.createFileWhenFull && eventsList.Count > settings.EventMaximum)
                        {
                            JsonParser.SaveRunning(); //Saves all current Events and starts again with 0 Events
                        }
                    }
                    else
                    {
                        savedEvents[e.eventName] = new List<Event>();
                        List<Event> eventsList = (List<Event>)savedEvents[e.eventName];
                        eventsList.Add(e);
                        savedEvents[e.eventName] = eventsList;
                    }
                }

                ///<summary>Removes every second element from a list and return the reduced list.</summary>
                public static List<Event> ReduceListData(List<Event> l)
                {
                    int oldLength = l.Count;
                    for (int i = 0; i < oldLength; i++)
                    {
                        if (i >= l.Count)
                        {
                            i = oldLength;
                        }
                        else
                        {
                            l.RemoveAt(i);
                        }
                    }
                    return l;
                }

                ///<summary>Sends all events to the JSON parser.</summary>
                public static void SendEvents()
                {
                    JsonParser.ReceiveEvents(savedEvents);
                }

                public static Hashtable GetEvents()
                {
                    return savedEvents;
                }

                ///<summary>Deletes all received events.</summary>
                public static void ClearEvents()
                {
                    savedEvents.Clear();
                    savedEvents = new Hashtable();
                }
            }
        }
    }
}
