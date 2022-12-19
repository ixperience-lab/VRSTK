using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

public class ChangeSpotLightReplay : MonoBehaviour
{
    [SerializeField]
    private List<GameObject> _spotLights;

    [SerializeField]
    private string _currentSpotLightColor; 

    public string CurrentSpotLightColor
    {
        get
        {
            return _currentSpotLightColor;
        }
        set
        {
            _currentSpotLightColor = value;
            Replay();
        }
    }

    [SerializeField]
    private float _currentSpotLightAngle;

    public float CurrentSpotLightAngle
    {
        get
        {
            return _currentSpotLightAngle;
        }
        set
        {
            _currentSpotLightAngle = value;
            //Replay();
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (TestStage.GetStarted())
        {
            string colorAsString = string.Format("{0};{1};{2};{3}", _spotLights[0].GetComponent<Light>().color.r,
                                                                    _spotLights[0].GetComponent<Light>().color.g,
                                                                    _spotLights[0].GetComponent<Light>().color.b,
                                                                    _spotLights[0].GetComponent<Light>().color.a);
            GetComponents<EventSender>()[1].SetEventValue("CurrentSpotLightColor_ChangeSpotLightReplay", colorAsString);
            GetComponents<EventSender>()[1].SetEventValue("CurrentSpotLightAngle_ChangeSpotLightReplay", _spotLights[0].GetComponent<Light>().spotAngle);
            GetComponents<EventSender>()[1].Deploy();
        }
    }

    private void Replay()
    {
        for (int i = 0; i < _spotLights.Count; i++)
        {
            _spotLights[i].GetComponent<Light>().spotAngle = _currentSpotLightAngle;
            if (_currentSpotLightColor != "")
                _spotLights[i].GetComponent<Light>().color = new Color(float.Parse(_currentSpotLightColor.Split(';')[0]), 
                                                                        float.Parse(_currentSpotLightColor.Split(';')[1]), 
                                                                        float.Parse(_currentSpotLightColor.Split(';')[2]), 
                                                                        float.Parse(_currentSpotLightColor.Split(';')[3]));
        }
    }
}
