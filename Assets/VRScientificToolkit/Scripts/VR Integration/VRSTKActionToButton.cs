using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Management;
using UnityEngine.XR.OpenXR.Input;

public class VRSTKActionToButton : VRSTKActionToControl
{
    //[SerializeField] private GameObject _gameObject = null;

    [SerializeField] private Color _normalColor = Color.red;

    [SerializeField] private Color _pressedColor = Color.green;

    private void Awake()
    {
        //if (_gameObject != null)
        //{
        //    _gameObject.SetActive(false);
        //    _gameObject.GetComponent<SpriteRenderer>().color = _normalColor;
        //}
        //gameObject.SetActive(false);
        //gameObject.GetComponent<SpriteRenderer>().color = _normalColor;
        gameObject.GetComponent<Renderer>().material.color = _normalColor;
    
    }

    protected override void OnActionStarted(InputAction.CallbackContext ctx)
    {
        //if (_gameObject != null)
        //    _gameObject.GetComponent<SpriteRenderer>().color = _pressedColor;
        //gameObject.GetComponent<SpriteRenderer>().color = _pressedColor;
        gameObject.GetComponent<Renderer>().material.color = _pressedColor;
        //Debug.Log("OnActionStarted Position: " + ctx.ReadValue<Vector2>());
    }

    protected override void OnActionCanceled(InputAction.CallbackContext ctx)
    {
        //if (_gameObject != null)
        //    _gameObject.GetComponent<SpriteRenderer>().color = _normalColor;
        //gameObject.GetComponent<SpriteRenderer>().color = _normalColor;
        gameObject.GetComponent<Renderer>().material.color = _normalColor;
        //Debug.Log("OnActionCanceled Position: " + ctx.ReadValue<Vector2>());
    }

    protected override void OnActionBound()
    {
        //if (_gameObject != null)
        //    _gameObject.SetActive(true);
        //gameObject.SetActive(true);
    }
}
