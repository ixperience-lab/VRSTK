using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Enable all input action mappings of an InputActionAsset, based on OpenXR plugin controller example impelementation</summary>
            public class InputActionMappingsEnabler : MonoBehaviour
            {
                [Tooltip("Input Action Asset")]
                [SerializeField] InputActionAsset _actionAsset;

                private void OnEnable()
                {
                    if (_actionAsset != null)
                        _actionAsset.Enable();
                }
            }
        }
    }
}
