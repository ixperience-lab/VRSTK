using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.XR.OpenXR.Input;
using VRSTK.Scripts.Telemetry;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Tracks all VR-Inputs using OpenXR and InputActionAsset and sends them out as Events for Tracking</summary>
            [RequireComponent(typeof(EventSender))]
            public class InputTracker : MonoBehaviour
            {
                [Tooltip("Input Action Mapping")]
                [SerializeField] InputActionAsset _inputActionMapping;

                private bool initialized = false;

                void Start()
                {

                }

                void Update()
                {
                    if (!initialized && _inputActionMapping.enabled)
                    {
                        foreach (InputAction inputAction in _inputActionMapping)
                        {
                            Debug.Log("InputAction name: " + inputAction.name);
                            inputAction.performed += OnPerform;
                        }
                        initialized = true;
                    }
                }

                private void OnPerform(InputAction.CallbackContext ctx)
                {
                    if (ctx.action.actionMap.ToString().ToLower().Contains("vrstk head")) return;

                    EventSender sender = GetComponent<EventSender>();

                    //Debug.Log("ctx.valueType: " + ctx.valueType);

                    sender.SetEventValue("Name", ctx.action.name);

                    if (ctx.valueType == typeof(System.Boolean))
                    {
                        sender.SetEventValue("boolValue", ctx.ReadValue<bool>());
                    }
                    if (ctx.action.type == InputActionType.Button)
                    {
                        sender.SetEventValue("buttonTriggertValue", ctx.ReadValueAsButton());
                    }
                    if (ctx.valueType == typeof(System.Int32))
                    {
                        sender.SetEventValue("integerValue", ctx.ReadValue<System.Int32>());
                    }
                    if (ctx.valueType == typeof(System.Double))
                    {
                        sender.SetEventValue("doubleValue", ctx.ReadValue<System.Double>());
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
                    if (ctx.valueType == typeof(UnityEngine.XR.OpenXR.Input.Haptic))
                    {
                        sender.SetEventValue("HapticValue", ctx.ReadValue<UnityEngine.XR.OpenXR.Input.Haptic>());
                    }
                    if (ctx.valueType == typeof(UnityEngine.XR.OpenXR.Input.Pose))
                    {
                        sender.SetEventValue("HapticValue", ctx.ReadValue<UnityEngine.XR.OpenXR.Input.Pose>());
                    }
                    sender.Deploy();
                }
            }
        }
    }
}
