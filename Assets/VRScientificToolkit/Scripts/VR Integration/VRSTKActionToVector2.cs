using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class VRSTKActionToVector2 : VRSTKActionToControl
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
        {
            _gameObjectReference.transform.localPosition = new Vector3(touchPosition2D.x, _gameObjectReference.transform.localPosition.y, touchPosition2D.y);
            //Debug.Log("gameObject.transform.position: " + gameObject.transform.position);
        }
    }
}
