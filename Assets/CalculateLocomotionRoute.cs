using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CalculateLocomotionRoute : MonoBehaviour
{
    public Vector3 _headsetPotion;
    public float _yOffset;

    private List<Vector3> _headsetPotions;

    public Vector3 HeadsetPotion
    {
        get { return _headsetPotion; }
        set
        {
            _headsetPotion = value;
            AddPoitToLineRenderer();
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        _headsetPotions = new List<Vector3>();
        _headsetPotions.Clear();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void AddPoitToLineRenderer()
    {
        _headsetPotions.Add(new Vector3(_headsetPotion.x, _headsetPotion.y - _yOffset, _headsetPotion.z));
        
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        
        Vector3[] headsetPotions = _headsetPotions.ToArray();
        lineRenderer.positionCount = headsetPotions.Length;
        lineRenderer.SetPositions(headsetPotions);
    }
}
