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
        RaycastHit hitPosition;
        float maxDistance = 2.0f;
        Physics.Raycast(_headsetPotion, Vector3.down, out hitPosition, maxDistance);
        Debug.Log("hitPosition: " + hitPosition.point);
        if (Physics.Raycast(_headsetPotion, Vector3.down, out hitPosition, maxDistance))
        {
            _headsetPotions.Add(new Vector3(hitPosition.point.x, hitPosition.point.y, hitPosition.point.z));

            LineRenderer lineRenderer = GetComponent<LineRenderer>();

            Vector3[] headsetPotions = _headsetPotions.ToArray();
            lineRenderer.positionCount = headsetPotions.Length;
            lineRenderer.SetPositions(headsetPotions);
        }
    }
}
