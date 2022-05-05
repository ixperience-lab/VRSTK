using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LocomotionRouteWithLinesPlayback : MonoBehaviour
{
    public Vector3 _headsetPotion;

    public Vector3 _hitPosition;

    public Vector3 _direction;

    public System.Single Duration;

    public Material _red;

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

    public Vector3 Direction
    {
        get { return _direction; }
        set { _direction = value; }
    }

    public Vector3 HitPosition
    {
        get { return _hitPosition; }
        set
        {
            _hitPosition = value;
            CreateLine();
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
        _headsetPotions.Add(_headsetPotion);
        
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        
        Vector3[] headsetPotions = _headsetPotions.ToArray();
        lineRenderer.positionCount = headsetPotions.Length;
        lineRenderer.SetPositions(headsetPotions);
    }

    void CreateLine()
    {
        //Vector3 headHit = (_hitPosition - _headsetPotion).normalized;
        //Vector3 secondPosition = _headsetPotion + (1.5f * headHit);
        Vector3 secondPosition = _hitPosition;

        //if (Duration > 0.5f)
        {
            GameObject eyeViewLine = new GameObject();
            eyeViewLine.transform.parent = transform;
            eyeViewLine.AddComponent<LineRenderer>();
            LineRenderer lr = eyeViewLine.GetComponent<LineRenderer>();
            lr.material = _red;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            //lr.SetColors(color, color);
            lr.SetWidth(0.01f, 0.01f);
            lr.SetPosition(0, _headsetPotion);
            lr.SetPosition(1, secondPosition);
        }
        //if (Duration > 0.5f)
        //    Debug.DrawLine(_headsetPotion, secondPoint, _red.color);
        //Gizmos.color = _red.color;
        //Gizmos.DrawLine(_headsetPotion, secondPoint);
    }
}
