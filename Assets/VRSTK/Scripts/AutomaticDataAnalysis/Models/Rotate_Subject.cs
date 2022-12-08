using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate_Subject : MonoBehaviour {

    float RotationSpeedOfAMouse = 10;


    // Use this for initialization
    void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
		
	}

    private void OnMouseDrag()
    {
        float x = Input.GetAxis("Mouse X") * RotationSpeedOfAMouse * Mathf.Deg2Rad;
        transform.RotateAround(Vector3.down, x);
        Debug.Log("Mousdrag");
    }
}