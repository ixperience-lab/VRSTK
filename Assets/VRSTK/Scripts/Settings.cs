using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace VRSTK
{
    namespace Scripts
    {
        ///<summary>Stores Project wide settings. Set values in resources folder</summary>
        public class Settings : ScriptableObject
        {

            public string jsonPath;
            [Tooltip("Maximum number of each type of event that will be stored. Value below 100.000 is suggested.")]
            public int EventMaximum;
            [Tooltip("When the maximum event number is reached, an event will be removed from the beginning for each new event added.")]
            [HideInInspector]
            public bool useSlidingWindow;
            [HideInInspector]
            public bool useDataReduction;
            [HideInInspector]
            public bool createFileWhenFull;
        }
    }
}
