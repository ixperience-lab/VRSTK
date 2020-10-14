using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace STK
{
    ///<summary>Script that stores all Gameobjects with a tracker to use in the playback module</summary>
    public class STKTrackedObjects : MonoBehaviour
    {

        public List<GameObject> trackedObjects;

        public void CheckForNullReferences() //Checks for objects which don't exist or don't have an Eventsender and removes them
        {
            foreach (GameObject g in trackedObjects)
            {
                if (g == null || g.GetComponent<STKEventSender>() == null)
                {
                    trackedObjects.Remove(g);
                }
            }
        }
    }
}
