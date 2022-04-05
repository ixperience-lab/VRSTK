using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;

namespace VRSTK
{
    namespace Scripts
    {
        namespace Playback
        {
            ///<summary>Script that stores all Gameobjects with a tracker to use in the playback module</summary>
            public class TrackedObjects : MonoBehaviour
            {

                public List<GameObject> trackedObjects;

                public void CheckForNullReferences() //Checks for objects which don't exist or don't have an Eventsender and removes them
                {
                    foreach (GameObject g in trackedObjects)
                    {
                        if (g == null || g.GetComponent<EventSender>() == null)
                        {
                            trackedObjects.Remove(g);
                        }
                    }
                }
            }
        }
    }
}
