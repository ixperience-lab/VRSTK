using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LineRendererRoute : MonoBehaviour
{
    [SerializeField]
    private Transform[] _controlPoints;
    
    private LineRenderer _lineRenderer;
    
    // Start is called before the first frame update
    void Start()
    {
        _lineRenderer = GetComponent<LineRenderer>();
        _lineRenderer.SetPosition(0, _controlPoints[0].position);
        _lineRenderer.SetPosition(1, _controlPoints[1].position);
        _lineRenderer.SetPosition(2, _controlPoints[2].position);
        _lineRenderer.SetPosition(3, _controlPoints[3].position);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
