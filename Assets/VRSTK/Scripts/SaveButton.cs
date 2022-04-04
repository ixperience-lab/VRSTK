using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;

namespace VRSTK
{
    namespace Scripts
    {
        ///<summary>Button on the test controller, which creates a JSON-file on click.</summary>
        public class SaveButton : MonoBehaviour
        {

            public void Onclick()
            {
                JsonParser.SaveRunning();
            }
        }
    }
}
