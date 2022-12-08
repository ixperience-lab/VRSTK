using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

public class ActiveModelsReplay : MonoBehaviour
{
    [SerializeField]
    public List<GameObject> _modelList = new List<GameObject>();

    [SerializeField]
    private string _currentActivatedModelName; // CurrentActivatedModelName_ActiveModelsReplay

    public string CurrentActivatedModelName
    {
        get 
        { 
            return _currentActivatedModelName; 
        }
        set
        {
            _currentActivatedModelName = value;
            //Debug.Log(value);
            Replay();
        }
    }

    private string _lastActivatedModelName = "";

    //private void OnValidate()
    //{
    //    // Only call properties during PlayMode since they might depend on runtime stuff
    //    //CurrentActivatedModelName = _currentActivatedModelName;
    //}

    // Start is called before the first frame update
    void Start()
    {
       
    }

    // Update is called once per frame
    void Update()
    {
        if (TestStage.GetStarted())
        {
            GetComponents<EventSender>()[1].SetEventValue("CurrentActivatedModelName_ActiveModelsReplay", GetComponent<ActivateModels>()._currentActivatedModelName);
            GetComponents<EventSender>()[1].Deploy();
        }
    }

    private void Replay()
    {
        if (_lastActivatedModelName != CurrentActivatedModelName)
            for (int i = 0; i < _modelList.Count; i++)
            {
                if (_modelList[i].name.Equals(_currentActivatedModelName))
                    _modelList[i].active = true;
                else
                    _modelList[i].active = false;
            }

        _lastActivatedModelName = CurrentActivatedModelName;
    }
}
