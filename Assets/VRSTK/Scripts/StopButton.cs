using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using VRSTK.Scripts.Telemetry;

namespace VRSTK
{
    namespace Scripts
    {
        ///<summary>Button which force-stops the experiment before all stages are finished.</summary>
        public class StopButton : MonoBehaviour
        {

            public void ForceStopTest()
            {
                EventReceiver.SendEvents();
                EventReceiver.ClearEvents();
                JsonParser.TestEnd();
                JsonParser.CreateFile();
                SceneManager.LoadScene(SceneManager.GetActiveScene().name);
            }
        }
    }
}