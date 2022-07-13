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

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (_effectActionActivated)
        {
            if (_isEffectActivatedDone && Time.deltaTime > _effectDurationTime && _spotLights != null)
            {
                for (int i = 0; i < _spotLights.Count; i++)
                {
                    _spotLights[i].GetComponent<Light>().spotAngle = 179.0f;
                    _spotLights[i].GetComponent<Light>().color = new Color(255f, 255f, 255f, 255f);
                }
                _effectActionActivated = false;
            }

            if (_objectToActivate != null && _spotLights != null && !_isEffectActivatedDone)
            {
                string activatedModelName = _objectToActivate.GetComponent<ActivateModels>()._currentActivatedModel.name;
                if (activatedModelName.Equals("Pose_Zombiegirl"))
                {
                    for (int i = 0; i < _spotLights.Count; i++)
                    {
                        _spotLights[i].GetComponent<Light>().spotAngle = 1.0f;
                        _spotLights[i].GetComponent<Light>().color = new Color(255f, 0f, 0f, 255f);
                    }
                    _isEffectActivatedDone = true;
                }
            }
        }
    }
}
