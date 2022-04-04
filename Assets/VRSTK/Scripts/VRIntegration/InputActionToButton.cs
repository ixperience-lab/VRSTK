using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR.Input;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Change color of a GameObject by using Button/press action, based on OpenXR plugin controller example impelementation</summary>
            public class InputActionToButton : InputActionToControl
            {
                [Tooltip("Default color")]
                [SerializeField] private Color _normalColor = Color.red;
                [Tooltip("Button press action color")]
                [SerializeField] private Color _pressedColor = Color.green;

                private void Awake()
                {
                    gameObject.GetComponent<Renderer>().material.color = _normalColor;
                }

                protected override void OnActionStarted(InputAction.CallbackContext ctx)
                {
                    gameObject.GetComponent<Renderer>().material.color = _pressedColor;
                }

                protected override void OnActionCanceled(InputAction.CallbackContext ctx)
                {
                    gameObject.GetComponent<Renderer>().material.color = _normalColor;
                }

                protected override void OnActionBound()
                {
                }
            }
        }
    }
}
