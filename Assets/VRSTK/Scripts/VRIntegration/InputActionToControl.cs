using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.OpenXR.Input;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Base class for using Actions to represent a control on a GameObject, based on OpenXR plugin controller example impelementation</summary>
            public class InputActionToControl : MonoBehaviour
            {
                [Tooltip("Action Reference that represents the control")]
                [SerializeField] private InputActionReference _actionReference = null;

                protected virtual void OnEnable()
                {
                    if (_actionReference == null || _actionReference.action == null)
                        return;

                    _actionReference.action.started += OnActionStarted;
                    _actionReference.action.performed += OnActionPerformed;
                    _actionReference.action.canceled += OnActionCanceled;

                    StartCoroutine(UpdateBinding());
                }

                protected virtual void OnDisable()
                {
                    if (_actionReference == null || _actionReference.action == null)
                        return;

                    _actionReference.action.started -= OnActionStarted;
                    _actionReference.action.performed -= OnActionPerformed;
                    _actionReference.action.canceled -= OnActionCanceled;
                }

                private IEnumerator UpdateBinding()
                {
                    while (isActiveAndEnabled)
                    {
                        if (_actionReference.action != null &&
                            _actionReference.action.controls.Count > 0 &&
                            _actionReference.action.controls[0].device != null &&
                            OpenXRInput.TryGetInputSourceName(_actionReference.action, 0, out var actionName, OpenXRInput.InputSourceNameFlags.Component, _actionReference.action.controls[0].device))
                        {
                            OnActionBound();
                            break;
                        }

                        yield return new WaitForSeconds(1.0f);
                    }
                }

                protected virtual void OnActionStarted(InputAction.CallbackContext ctx) { }

                protected virtual void OnActionPerformed(InputAction.CallbackContext ctx) { }

                protected virtual void OnActionCanceled(InputAction.CallbackContext ctx) { }

                protected virtual void OnActionBound() { }
            }
        }
    }
}
