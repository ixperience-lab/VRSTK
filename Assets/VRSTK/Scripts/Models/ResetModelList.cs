using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ResetModelList : MonoBehaviour
{
    public GameObject _objectToActivate;
    // Start is called before the first frame update
    void Start()
    {
        _objectToActivate.active = true;
        _objectToActivate.GetComponent<ActivateModels>().ResetModels();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
