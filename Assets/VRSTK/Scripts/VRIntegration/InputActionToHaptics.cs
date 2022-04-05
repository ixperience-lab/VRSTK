using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.OpenXR.Input;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Enable Haptic action for an Action reference, based on OpenXR plugin controller example impelementation</summary>
            public class InputActionToHaptics : MonoBehaviour
            {
                [Tooltip("Input Action reference")]
                [SerializeField] private InputActionReference _action;
                [Tooltip("Input Action Haptic reference")]
                [SerializeField] private InputActionReference _hapticAction;
                public float _amplitude = 1.0f;
                public float _duration = 0.1f;
                public float _frequency = 0.0f;

                private void Start()
                {
                    if (_action == null || _hapticAction == null)
                        return;

                    _action.action.Enable();
                    _hapticAction.action.Enable();
                    _action.action.performed += OnPerform;
                }

                private void OnPerform(InputAction.CallbackContext ctx)
                {
                    InputControl control = _action.action.activeControl;
                    if (null == control)
                        return;

                    OpenXRInput.SendHapticImpulse(_hapticAction.action, _amplitude, _frequency, _duration, control.device);
                }
            }
        }
    }
}
