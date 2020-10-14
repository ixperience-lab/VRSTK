using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
namespace STK
{
    ///<summary>Button which force-stops the experiment before all stages are finished.</summary>
    public class STKStopButton : MonoBehaviour
    {

        public void ForceStopTest()
        {
            STKEventReceiver.SendEvents();
            STKEventReceiver.ClearEvents();
            STKJsonParser.TestEnd();
            STKJsonParser.CreateFile();
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }
    }
}
