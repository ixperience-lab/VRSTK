using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeSpotLightAngleAndColor : MonoBehaviour
{
    [SerializeField]
    private bool _effectActionActivated = false;

    [SerializeField]
    private float _effectDurationTime = 1.0f;

    [SerializeField]
    private bool _isEffectActivatedDone = false;

    [SerializeField]
    private List<GameObject> _spotLights;

    [SerializeField]
    private GameObject _objectToActivate;

    private float _startTime = 0.0f;
    private float _diffTime = 0.0f;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {   
        if (_effectActionActivated)
        {
            //Debug.Log("DateTime.Now.Second : " + DateTime.Now.Second.ToString());
            //_diffTime += Time.deltaTime;
            //float duration = _diffTime - _startTime;
            //Debug.Log("_diffTime : " + _diffTime);
            if (_isEffectActivatedDone)
            {
                _diffTime += Time.deltaTime;
                if (_diffTime > _effectDurationTime && _spotLights != null)
                {    //Debug.Log("_effectActionActivated = true");
                    for (int i = 0; i < _spotLights.Count; i++)
                    {
                        _spotLights[i].GetComponent<Light>().spotAngle = 179.0f;
                        _spotLights[i].GetComponent<Light>().color = new Color(1.0f, 1.0f, 1.0f, 1.0f);
                    }
                    //Debug.Log("_effectActionActivated = false");
                    _effectActionActivated = false;
                }
            }

            if (_objectToActivate != null && _spotLights != null && !_isEffectActivatedDone)
            {
                if (_objectToActivate.GetComponent<ActivateModels>()._currentActivatedModel != null)
                {
                    string activatedModelName = _objectToActivate.GetComponent<ActivateModels>()._currentActivatedModel.name;

                    if (activatedModelName.Equals("Pose_Zombiegirl"))
                    {
                        for (int i = 0; i < _spotLights.Count; i++)
                        {
                            _spotLights[i].GetComponent<Light>().spotAngle = 1.0f;
                            _spotLights[i].GetComponent<Light>().color = new Color(1.0f, 0f, 0f, 1.0f);
                        }
                        //_startTime = Time.deltaTime;
                        //Debug.Log("_isEffectActivatedDone = true");
                        _isEffectActivatedDone = true;
                    }
                }
            }
        }
    }
}
