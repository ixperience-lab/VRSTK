using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

public class PageConcatenationToAvatarSelection : MonoBehaviour
{
    [SerializeField]
    public GameObject _objectToActivate;
    
    [SerializeField]
    public string _activatedModelName = "";
    
    [SerializeField]
    public int _currenSelectedtIndex = 0;

    // Start is called before the first frame update
    void Start()
    {
        if (TestStage.GetStarted())
        {
            GetSelectedModelInformations();
        }
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void GetSelectedModelInformations()
    {
        if (_objectToActivate != null)
        {
            _activatedModelName = _objectToActivate.GetComponent<ActivateModels>()._currentActivatedModel.name;
            _currenSelectedtIndex = _objectToActivate.GetComponent<ActivateModels>()._currenSelectedtIndex;
        }
    }
}
