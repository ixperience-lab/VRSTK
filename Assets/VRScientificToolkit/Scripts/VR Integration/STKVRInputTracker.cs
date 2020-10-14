using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Valve.VR;

namespace STK
{
    ///<summary>Tracks all VR-Inputs using SteamVRInput and sends them out as Events for Tracking</summary>
    [RequireComponent(typeof(STKEventSender))]
    public class STKVRInputTracker : MonoBehaviour
    {

        private bool initialized = false;

        void Start()
        {
        }


        void Update()
        {
            if (!initialized && SteamVR_Input.initialized) //Once Input is initialized, check for all Input Actions and create Listeners for them
            {
                foreach (SteamVR_Action_Boolean a in SteamVR_Input.actionsBoolean)
                {
                    Debug.Log(a);
                    a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
                }
                foreach (SteamVR_Action_Single a in SteamVR_Input.actionsSingle)
                {
                    Debug.Log(a);
                    a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
                }
                foreach (SteamVR_Action_Vector2 a in SteamVR_Input.actionsVector2)
                {
                    Debug.Log(a);
                    a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
                }
                foreach (SteamVR_Action_Vector3 a in SteamVR_Input.actionsVector3)
                {
                    Debug.Log(a);
                    a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
                }
                initialized = true;
            }
        }

        void OnChange(SteamVR_Action_In action) //Creates and deploys an Event on any Input action
        {
            STKEventSender sender = GetComponent<STKEventSender>();
            sender.SetEventValue("Name", action.GetShortName());
            if (action.GetType() == typeof(SteamVR_Action_Boolean))
            {
                SteamVR_Action_Boolean v = (SteamVR_Action_Boolean)action;
                sender.SetEventValue("boolValue", v.GetState(SteamVR_Input_Sources.Any));
            }
            else if (action.GetType() == typeof(SteamVR_Action_Single))
            {
                SteamVR_Action_Single v = (SteamVR_Action_Single)action;
                sender.SetEventValue("singleValue", v.GetAxis(SteamVR_Input_Sources.Any));
            }
            else if (action.GetType() == typeof(SteamVR_Action_Vector2))
            {
                SteamVR_Action_Vector2 v = (SteamVR_Action_Vector2)action;
                sender.SetEventValue("vector2Value", v.GetAxis(SteamVR_Input_Sources.Any));
            }
            else if (action.GetType() == typeof(SteamVR_Action_Vector3))
            {
                SteamVR_Action_Vector3 v = (SteamVR_Action_Vector3)action;
                sender.SetEventValue("vector3Value", v.GetAxis(SteamVR_Input_Sources.Any));
            }
            sender.Deploy();
        }
    }
}
