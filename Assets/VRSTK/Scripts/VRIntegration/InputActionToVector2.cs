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
            ///<summary>Change local position of a GameObject by using Axis coordinates action (type of Vector2), based on OpenXR plugin controller example impelementation</summary>
            public class InputActionToVector2 : InputActionToControl
            {
                [Tooltip("GameObject Reference that represents the control")]
                [SerializeField] private GameObject _gameObjectReference = null;
                protected override void OnActionPerformed(InputAction.CallbackContext ctx) => UpdateAction(ctx);

                protected override void OnActionStarted(InputAction.CallbackContext ctx) => UpdateAction(ctx);

                protected override void OnActionCanceled(InputAction.CallbackContext ctx) => UpdateAction(ctx);

                private void UpdateAction(InputAction.CallbackContext ctx)
                {
                    Vector2 touchPosition2D = (ctx.ReadValue<Vector2>()) * 0.5f;
                    if (touchPosition2D.x < 1 && touchPosition2D.x > -1 && touchPosition2D.y < 1 && touchPosition2D.y > -1)
                        _gameObjectReference.transform.localPosition = new Vector3(touchPosition2D.x, _gameObjectReference.transform.localPosition.y, touchPosition2D.y);
                }
            }
        }
    }
}
