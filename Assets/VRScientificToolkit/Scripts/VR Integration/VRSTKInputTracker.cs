using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.XR.OpenXR.Input;
//using Valve.VR;

namespace STK
{
    ///<summary>Tracks all VR-Inputs using SteamVRInput and sends them out as Events for Tracking</summary>
    [RequireComponent(typeof(STKEventSender))]
    public class VRSTKInputTracker : MonoBehaviour
    {
        [Tooltip("Input Action Mapping")]
        [SerializeField] InputActionAsset _inputActionMapping;

        private bool initialized = false;

        void Start()
        {
            
        }


        void Update()
        {
            if(!initialized && _inputActionMapping.enabled)
            {
                foreach(InputAction inputAction in _inputActionMapping)
                {
                    Debug.Log("InputAction name: " + inputAction.name);
                    inputAction.performed += OnPerform;
                }
                initialized = true;
            }

            //if (!initialized && SteamVR_Input.initialized) //Once Input is initialized, check for all Input Actions and create Listeners for them
            //{
            //    foreach (SteamVR_Action_Boolean a in SteamVR_Input.actionsBoolean)
            //    {
            //        Debug.Log(a);
            //        a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
            //    }
            //    foreach (SteamVR_Action_Single a in SteamVR_Input.actionsSingle)
            //    {
            //        Debug.Log(a);
            //        a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
            //    }
            //    foreach (SteamVR_Action_Vector2 a in SteamVR_Input.actionsVector2)
            //    {
            //        Debug.Log(a);
            //        a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
            //    }
            //    foreach (SteamVR_Action_Vector3 a in SteamVR_Input.actionsVector3)
            //    {
            //        Debug.Log(a);
            //        a.AddOnChangeListener(OnChange, SteamVR_Input_Sources.Any);
            //    }
            //    initialized = true;
            //}
        }

        private void OnPerform(InputAction.CallbackContext ctx)
        {
            if (ctx.action.name.Contains("hmd") && ctx.action.name.Contains("Eye")) return;

            STKEventSender sender = GetComponent<STKEventSender>();

            Debug.Log("ctx.valueType: " + ctx.valueType);
            
            sender.SetEventValue("Name", ctx.action.name);
            
            if (ctx.valueType == typeof(System.Boolean))
            {
                sender.SetEventValue("boolValue", ctx.ReadValue<bool>());
            }
            if (ctx.valueType == typeof(Button))
            {
                sender.SetEventValue("buttonBoolValue", ctx.ReadValueAsButton());
            }
            if (ctx.valueType == typeof(System.Single))
            {
                sender.SetEventValue("singleValue", ctx.ReadValue<System.Single>());
            }
            if (ctx.valueType == typeof(Vector2))
            {
                sender.SetEventValue("vector2Value", ctx.ReadValue<Vector2>());
            }
            if (ctx.valueType == typeof(Vector3))
            {
                sender.SetEventValue("vector3Value", ctx.ReadValue<Vector3>());
            }
            if (ctx.valueType == typeof(Quaternion))
            {
                sender.SetEventValue("quaternionValue", ctx.ReadValue<Quaternion>());
            }

            sender.Deploy();
        }

        //private void OnChange(SteamVR_Action_Boolean actionIn, SteamVR_Input_Sources inputSource, bool newState)
        //{
        //    STKEventSender sender = GetComponent<STKEventSender>();
        //    sender.SetEventValue("Name", actionIn.GetShortName());
        //    if (actionIn.GetType() == typeof(SteamVR_Action_Boolean))
        //    {
        //        SteamVR_Action_Boolean v = (SteamVR_Action_Boolean)actionIn;
        //        sender.SetEventValue("boolValue", v.GetState(SteamVR_Input_Sources.Any));
        //    }
        //    sender.Deploy();
        //}

        //private void OnChange(SteamVR_Action_Single actionIn, SteamVR_Input_Sources fromSource, float newAxis, float newDelta)
        //{
        //    STKEventSender sender = GetComponent<STKEventSender>();
        //    sender.SetEventValue("Name", actionIn.GetShortName());
        //    if (actionIn.GetType() == typeof(SteamVR_Action_Single))
        //    {
        //        SteamVR_Action_Single v = (SteamVR_Action_Single)actionIn;
        //        sender.SetEventValue("singleValue", v.GetAxis(SteamVR_Input_Sources.Any));
        //    }
        //    sender.Deploy();
        //}

        //private void OnChange(SteamVR_Action_Vector2 actionIn, SteamVR_Input_Sources fromSource, Vector2 axis, Vector2 delta)
        //{
        //    STKEventSender sender = GetComponent<STKEventSender>();
        //    sender.SetEventValue("Name", actionIn.GetShortName());
        //    if (actionIn.GetType() == typeof(SteamVR_Action_Vector2))
        //    {
        //        SteamVR_Action_Vector2 v = (SteamVR_Action_Vector2)actionIn;
        //        sender.SetEventValue("vector2Value", v.GetAxis(SteamVR_Input_Sources.Any));
        //    }
        //    sender.Deploy();
        //}

        //private void OnChange(SteamVR_Action_Vector3 actionIn, SteamVR_Input_Sources fromSource, Vector3 axis, Vector3 delta)
        //{
        //    STKEventSender sender = GetComponent<STKEventSender>();
        //    sender.SetEventValue("Name", actionIn.GetShortName());
        //    if (actionIn.GetType() == typeof(SteamVR_Action_Vector3))
        //    {
        //        SteamVR_Action_Vector3 v = (SteamVR_Action_Vector3)actionIn;
        //        sender.SetEventValue("vector3Value", v.GetAxis(SteamVR_Input_Sources.Any));
        //    }
        //    sender.Deploy();
        //}
    }
}
