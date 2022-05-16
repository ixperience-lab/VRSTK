using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OnRailLocomotionRoutesWithAnimationCurves : MonoBehaviour
{
    public Transform _target;

    public Transform [] _controlPoints;
    //public Transform _controlPoint2;

    public AnimationCurve []  _lerpCurves;

    public Vector3 [] _lerpOffsets;

    public float _lerpTime = 30f;

    private float _timer = 0f;
    private float _currentTimer = 0f;
    private float _deltaTimeForEachPair = 0f;
    private float _totalDeltaTimer = 0f;
    
    private int _lerpIndex = 0;
    private int _controlPointIndex1 = 0;
    private int _controlPointIndex2 = 0;

    private void Start()
    {
        _deltaTimeForEachPair = 0f;

        if (_controlPoints.Length > 1)
        {
            _deltaTimeForEachPair = _lerpTime / (_controlPoints.Length/2);
            _totalDeltaTimer += _deltaTimeForEachPair;
        }
        _lerpIndex = 0;
        _controlPointIndex1 = 0;
        _controlPointIndex2 = 1;

    }

    void Update()
    {
        _timer += Time.deltaTime;
        if (_timer <= _totalDeltaTimer && _controlPointIndex2 <= _controlPoints.Length - 1)
        {
            LerpPosition(_lerpIndex, _controlPointIndex1, _controlPointIndex2);
        }
        else if(_controlPointIndex2 <= _controlPoints.Length - 1)
        { 
            _lerpIndex += 1;
            _controlPointIndex1 += 2;
            _controlPointIndex2 += 2;
            _totalDeltaTimer += _deltaTimeForEachPair;
        }

    }

    void LerpPosition( int lerpIndex, int controlPointIndex1, int controlPointIndex2)
    {
        if (_timer > _lerpTime)
            _timer = _lerpTime;

        float lerpRatio = _timer / _lerpTime;

        Vector3 positionOffset = _lerpCurves[lerpIndex].Evaluate(lerpRatio) * _lerpOffsets[0];

        _target.position = Vector3.Lerp(_controlPoints[controlPointIndex1].position, _controlPoints[controlPointIndex2].position, lerpRatio) + positionOffset;
    }
}
