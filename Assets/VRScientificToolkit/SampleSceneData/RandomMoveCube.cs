using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RandomMoveCube : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	
	// Update is called once per frame
	void Update () {
        transform.Translate(new Vector3(Random.Range(-0.01f, 0.01f), 0, Random.Range(-0.01f, 0.01f)));
        transform.Rotate(new Vector3(0, 1, 0));
	}
}
