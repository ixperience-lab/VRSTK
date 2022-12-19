using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using VRSTK.Scripts.TestControl;

public class OnRailLocomotionRoutes : MonoBehaviour
{
    [SerializeField]
    private Transform _xROrigin;

    [SerializeField]
    private Transform [] _routes;

    [SerializeField]
    private float _speedModifier;

    private int _routeNumber;

    private float _tParam;

    private bool _coroutineAllowed;

    private float _time;

    private float _xROriginYPosition;

    // Start is called before the first frame update
    void Start()
    {
        _routeNumber = 0;
        _xROriginYPosition = _xROrigin.position.y;
        _xROrigin.position = new Vector3(_routes[_routeNumber].GetChild(0).position.x, _xROrigin.position.y, _routes[_routeNumber].GetChild(0).position.z);
        _tParam = 0f;
        _coroutineAllowed = true;
    }

    // Update is called once per frame
    void Update()
    {
        _time += Time.deltaTime;
        bool testTageStarted = TestStage.GetStarted();
        if (_coroutineAllowed && testTageStarted && _routeNumber < _routes.Length)
            StartCoroutine(OnRailLocomotion(_routeNumber));
    }

    private IEnumerator OnRailLocomotion(int routeNumber)
    {
        _coroutineAllowed = false;

        Vector3 cp0 = new Vector3(_routes[routeNumber].GetChild(0).position.x, _routes[routeNumber].GetChild(0).position.y + _xROriginYPosition, _routes[routeNumber].GetChild(0).position.z);
        Vector3 cp1 = new Vector3(_routes[routeNumber].GetChild(1).position.x, _routes[routeNumber].GetChild(1).position.y + _xROriginYPosition, _routes[routeNumber].GetChild(1).position.z);
        Vector3 cp2 = new Vector3(_routes[routeNumber].GetChild(2).position.x, _routes[routeNumber].GetChild(2).position.y + _xROriginYPosition, _routes[routeNumber].GetChild(2).position.z);
        Vector3 cp3 = new Vector3(_routes[routeNumber].GetChild(3).position.x, _routes[routeNumber].GetChild(3).position.y + _xROriginYPosition, _routes[routeNumber].GetChild(3).position.z);

        while (_tParam < 1)
        {
            _tParam += Time.deltaTime * _speedModifier;
            Vector3 newPosition = Mathf.Pow(1 - _tParam, 3) * cp0 + 3 * _tParam * Mathf.Pow(1 - _tParam, 2) * cp1 + 3 * Mathf.Pow(_tParam, 2) * (1 - _tParam) * cp2 + Mathf.Pow(_tParam, 3) * cp3;

            _xROrigin.position = new Vector3(newPosition.x, newPosition.y, newPosition.z);

            yield return new WaitForEndOfFrame();
        }

        _tParam = 0f;

        _routeNumber += 1;

        if (_routeNumber > _routes.Length)
            _coroutineAllowed = false;
        else
            _coroutineAllowed = true;
    }

}
