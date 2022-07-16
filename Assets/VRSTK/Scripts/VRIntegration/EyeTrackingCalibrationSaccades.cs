using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ViveSR.anipal.Eye;
using ViveSR.anipal;
using ViveSR;
using System.Runtime.InteropServices;
using System;

public class EyeTrackingCalibrationSaccades : MonoBehaviour
{
    public List<Vector3> _saccads;
    public List<Vector3> _fixations;
    public List<Vector3> _currentfixations;

    private float _currentTime, _lastTime = 0f;

    public int _saccadeCounter = 0;
    public int _fixationCounter = 0;
    public float _saccadeVelocityThreshold = 300f;
    //public bool _isEyeDataCallbackFunctionRegistered = false;
    public bool _isUserNeedCalibration;
    public bool _calibrationResult;
    public float _measuredDifferceTime = 0;
    public float _measuredVisualAngle = 0f;
    public float _measuredVelocity = 0;
    //public int _countCallback = 0;

    [SerializeField]
    public float _leftEyeOpenness, _rightEyeOpenness;

    [SerializeField]
    public float _leftPupilDiameter, _rightPupilDiameter;

    [SerializeField]
    public Vector2 _leftPupilPositionInSensorArea, _rightPupilPositionInSensorArea;

    [SerializeField]
    public Vector3 _leftGazeDirectionNormalized, _rightGazeDirectionNormalized;

    [SerializeField]
    public Vector3 _leftGazeDirectionNormalizedTranslatedToWorldSpace, _rightGazeDirectionNormalizedTranslatedToWorldSpace;

    [SerializeField]
    public Vector3 _combinedGazeDirectionNormalized, _combinedGazeDirectionNormalizedTranslatedToWorldSpace;

    [SerializeField]
    public int _sRanipalFrameSequence;
    
    [SerializeField]
    public UInt64 _leftEyeDataValidataBitMask, _rightEyeDataValidataBitMask;
    
    [SerializeField]
    public Vector3 _leftGazeOrigin_mm, _rightGazeOrigin_mm;
    
    //[SerializeField]
    //public float _leftEyeExpressionData_EyeFrown, _rightEyeExpressionData_EyeFrown;

    //[SerializeField]
    //public float _leftEyeExpressionData_EyeSqueeze, _rightEyeExpressionData_EyeSqueeze;

    //[SerializeField]
    //public float _leftEyeExpressionData_EyeWide, _rightEyeExpressionData_EyeWide;

    [SerializeField]
    public double _gazeSensitiveFactor;

    [SerializeField]
    public float _convergenceDistance_mm;

    [SerializeField]
    public bool _convergenceDistanceValidity;

    private EyeData _eyeData = new EyeData();//private EyeData_v2 _eyeDataV2 = new EyeData_v2();
    private ViveSR.Error _lastError;
    private Vector3 _meanHeadPosition;
    public List<Vector3> _headPositions;
    public Material _material;
    private List<Vector3> tempSaccades;

    // Start is called before the first frame update
    void Start()
    {
        //Invoke("CheckAndInitSRanipal", 0.5f);

        _saccads = new List<Vector3>();
        _fixations = new List<Vector3>();
        _currentfixations = new List<Vector3>();
        _headPositions = new List<Vector3>();
        tempSaccades = new List<Vector3>();

        //Initialize|Instancing off eye tracking 
        if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING)
        {
            Debug.LogWarning("SRanipal not running (Status " + SRanipal_Eye_Framework.Status + "). Trying to (re)initialise");
            var sranipal = SRanipal_Eye_Framework.Instance;
            if (sranipal == null)
            {
                //The Framework script should be included in the scene
                //If not, as a fallback, create a component
                Debug.LogWarning("SRanipal_Eye_Framework should be included in world.");
                sranipal = gameObject.AddComponent<SRanipal_Eye_Framework>();
            }
            sranipal.StartFramework();
        }

        //if (!SRanipal_Eye_v2.LaunchEyeCalibration()) Calibration();
    }

    // Update is called once per frame
    void Update()
    {
        var error = SRanipal_Eye_API.GetEyeData(ref _eyeData); //SRanipal_Eye_API.GetEyeData_v2(ref _eyeDataV2);

        var newData = SRanipal_Eye.GetVerboseData(out _eyeData.verbose_data);

        //Show error only once and not every frame
        if (error != ViveSR.Error.WORK && _lastError != error)
        {
            Debug.LogError("An error happened: " + error.ToString());
        }
        _lastError = error;


        //No new data received from camera sensor - skip step
        if (!newData)
            return;

        //_currentTime = Time.deltaTime;

        _currentTime = (float) _eyeData.timestamp / 1000f;

        EyeParameter eyeParameter = new EyeParameter();
        SRanipal_Eye_API.GetEyeParameter(ref eyeParameter);

        var leftEye = _eyeData.verbose_data.left.gaze_direction_normalized;
        var rightEye = _eyeData.verbose_data.right.gaze_direction_normalized;
        
        
        
        _sRanipalFrameSequence = _eyeData.frame_sequence;
        _leftEyeDataValidataBitMask = _eyeData.verbose_data.left.eye_data_validata_bit_mask;
        _rightEyeDataValidataBitMask = _eyeData.verbose_data.right.eye_data_validata_bit_mask;
        _leftEyeOpenness = _eyeData.verbose_data.left.eye_openness;
        _rightEyeOpenness = _eyeData.verbose_data.right.eye_openness;
        _leftPupilDiameter = _eyeData.verbose_data.left.pupil_diameter_mm;
        _rightPupilDiameter = _eyeData.verbose_data.right.pupil_diameter_mm;
        _leftPupilPositionInSensorArea = _eyeData.verbose_data.left.pupil_position_in_sensor_area;
        _rightPupilPositionInSensorArea = _eyeData.verbose_data.right.pupil_position_in_sensor_area;
        _leftGazeOrigin_mm = _eyeData.verbose_data.left.gaze_origin_mm;
        _rightGazeOrigin_mm = _eyeData.verbose_data.right.gaze_origin_mm;
        _leftGazeDirectionNormalized = _eyeData.verbose_data.left.gaze_direction_normalized;                                               
        _rightGazeDirectionNormalized = _eyeData.verbose_data.right.gaze_direction_normalized;
        _gazeSensitiveFactor = eyeParameter.gaze_ray_parameter.sensitive_factor;
        //_leftEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.left.eye_frown;
        //_rightEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.right.eye_frown;
        //_leftEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.left.eye_squeeze;
        //_rightEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.right.eye_squeeze;
        //_leftEyeExpressionData_EyeWide = _eyeDataV2.expression_data.left.eye_wide;
        //_rightEyeExpressionData_EyeWide = _eyeDataV2.expression_data.right.eye_wide;
        _combinedGazeDirectionNormalized = _eyeData.verbose_data.combined.eye_data.gaze_direction_normalized;
        _convergenceDistanceValidity = _eyeData.verbose_data.combined.convergence_distance_validity;
        _convergenceDistance_mm = _eyeData.verbose_data.combined.convergence_distance_mm;

        CalculateWorldSpace(_leftGazeDirectionNormalized, ref _leftGazeDirectionNormalizedTranslatedToWorldSpace);
        CalculateWorldSpace(_rightGazeDirectionNormalized, ref _rightGazeDirectionNormalizedTranslatedToWorldSpace);
        CalculateWorldSpace(_combinedGazeDirectionNormalized, ref _combinedGazeDirectionNormalizedTranslatedToWorldSpace);

        Vector3 v = GazeIntersectionPoint(_combinedGazeDirectionNormalizedTranslatedToWorldSpace);

        _measuredDifferceTime = 0f;
        if (_lastTime != 0)
            _measuredDifferceTime = _currentTime - _lastTime;

        CalculateSaccades(v, _measuredDifferceTime);
        
        _lastTime = _currentTime;
    }

    //  Check if SRanipal_Eye works properly.
    void CheckAndInitSRanipal()
    {
        Error errorCode = SRanipal_API.Initial(SRanipal_Eye_v2.ANIPAL_TYPE_EYE_V2, IntPtr.Zero);

        if (errorCode == Error.WORK)
            Debug.Log("[SRanipal] Initial Eye v2: " + errorCode);
        else
        {
            Debug.LogError("[SRanipal] Initial Eye v2: " + errorCode);
            Debug.LogError("Check SRanipal runtime and use prefab properl! After this restart the scene again");

            if (UnityEditor.EditorApplication.isPlaying)
                UnityEditor.EditorApplication.isPlaying = false;    // Stops Unity editor.
        }
    }

    //  Calibration is performed if the calibration is necessary.
    void Calibration()
    {
        SRanipal_Eye_API.IsUserNeedCalibration(ref _isUserNeedCalibration);

        if (_isUserNeedCalibration)
        {
            Debug.LogWarning("Somthing went wrong in the calibration process the user need to recalibrate!");
            if (SRanipal_Eye_v2.LaunchEyeCalibration())
                Debug.Log("Calibration is done successfully.");
            else
            {
                Debug.Log("Calibration is failed. Restart the scene again");
                if (UnityEditor.EditorApplication.isPlaying)
                    UnityEditor.EditorApplication.isPlaying = false;    // Stops Unity editor.
            }
        }

    }

    private void CalculateWorldSpace(Vector3 direction, ref Vector3 gazeDirectionNormalizedTranslatedToWorldSpace)
    {
        // The direction from sr_anipal is in "raw" world space
        // So no translation or rotation of the head is taken in account
        // This will translate it to the location and rotation of the world space
        direction = transform.TransformDirection(direction);

        // The data we get from sr_anipal is wrongly coded
        // When looking up right, we get the direction up left
        // Instead of down left, we get down right
        // So we switch to 2d space (using x and z but not the height y)
        // Then we get the angle between the cameras lookat and the eye lookat
        // We negate the angle, multipliying by two (lookat -> center -> lookat) and
        // use the quaternion transform to get the "real" data.
        var eyeDirection2 = new Vector2(direction.x, direction.z);
        var headDirection2 = new Vector2(transform.forward.x, transform.forward.z);
        gazeDirectionNormalizedTranslatedToWorldSpace = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * direction;
    }

    private Vector3 GazeIntersectionPoint(Vector3 direction)
    {
        int layerMask = 1 << 8;
        layerMask = ~layerMask;
        Vector3 hitPoint = direction.normalized * 100;
        string hitlog = "No hit on an Object Collider!";
        RaycastHit hitObjects;
        if (Physics.Raycast(transform.position, direction, out hitObjects, Mathf.Infinity, layerMask))
        {
            hitlog = "Hit the Collider of an Object, with direction:" + direction + " and hitPoint:" + hitObjects.point;
            //When hit collider: Use collider as hitpoint
            hitPoint = hitObjects.point;
        }
       
        //Debug.Log(hitlog);
        return hitPoint;
    }

    private Vector3 CalculateMeanVektor(List<Vector3> vectorList)
    {
        return vectorList[0] + 0.5f * (vectorList[0] - vectorList[1]);
    }

    private void CalculateSaccades(Vector3 gazeVector, float timeDiff)
    {
        if (_currentfixations.Count < 2)
        {
            _headPositions.Add(transform.position);
            _currentfixations.Add(gazeVector);
            return;
        }

        //_meanHeadPosition = CalculateMeanVektor(_headPositions);

        Vector3 currentfixations_v0 = new Vector3(_currentfixations[0].x, _currentfixations[0].y, _currentfixations[0].z);// - _meanHeadPosition;
        Vector3 currentfixations_v1 = new Vector3(_currentfixations[1].x, _currentfixations[1].y, _currentfixations[1].z); ;// - _meanHeadPosition;
        //float visualAngle = CalculateVisualAngle(_currentfixations[0], _currentfixations[1]);
        float visualAngle = Mathf.Acos(Vector3.Dot(currentfixations_v0, currentfixations_v1) / (currentfixations_v0.magnitude * currentfixations_v1.magnitude)) * Mathf.Rad2Deg;

        if (timeDiff == 0) timeDiff = 1f;
        float gazeVelocity = visualAngle / timeDiff;

        _measuredVisualAngle = visualAngle;
        _measuredVelocity = gazeVelocity;

        bool isSaccade = false;

        if (gazeVelocity > _saccadeVelocityThreshold)
        {
            _saccads.Clear();
            _fixations.Clear();
            
            //int counter = GetComponent<LineRenderer>().positionCount;
            //if (counter == 0) counter = 1;
            //GetComponent<LineRenderer>().SetPosition((counter - 1), _currentfixations[0]);
            //GetComponent<LineRenderer>().SetPosition(counter, _currentfixations[1]);
            _saccads.Add(new Vector3(currentfixations_v0.x, currentfixations_v0.y, currentfixations_v0.z));
            _saccads.Add(new Vector3(currentfixations_v1.x, currentfixations_v1.y, currentfixations_v1.z));
            //tempSaccades.Add(new Vector3(currentfixations_v0.x, currentfixations_v0.y, currentfixations_v0.z));
            //tempSaccades.Add(new Vector3(currentfixations_v1.x, currentfixations_v1.y, currentfixations_v1.z));
            _saccadeCounter++;
            //GetComponent<LineRenderer>().positionCount = tempSaccades.Count;
            //GetComponent<LineRenderer>().SetPositions(tempSaccades.ToArray());
            isSaccade = true;
        }



        //for (int i = 0; i < _currentfixations.Count; i++)
            if (!isSaccade)
            {
                //GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                //sphere.name = "TestSphere_" + currentfixations_v0.magnitude;
                //sphere.transform.position = new Vector3(currentfixations_v0.x, currentfixations_v0.y, currentfixations_v0.z);
                //sphere.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
                //sphere.GetComponent<MeshRenderer>().receiveShadows = false;
                //sphere.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                //Renderer rend = sphere.GetComponent<Renderer>();
                //rend.material = _material;

                //sphere.transform.parent = transform;

                //_fixations.Add(new Vector3(currentfixations_v0.x, currentfixations_v0.y, currentfixations_v0.z));
                _fixationCounter++;

                //sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                //sphere.name = "TestSphere_" + currentfixations_v1.magnitude;
                //sphere.transform.position = new Vector3(currentfixations_v1.x, currentfixations_v1.y, currentfixations_v1.z);
                //sphere.transform.localScale = new Vector3(0.01f, 0.01f, 0.01f);
                //sphere.GetComponent<MeshRenderer>().receiveShadows = false;
                //sphere.GetComponent<MeshRenderer>().shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                //rend = sphere.GetComponent<Renderer>();
                //rend.material = _material;

                //sphere.transform.parent = transform;

                //_fixations.Add(new Vector3(currentfixations_v1.x, currentfixations_v1.y, currentfixations_v1.z));
                _fixationCounter++;
            }

        Vector3 tempCurrentFixation = _currentfixations[1];
        _currentfixations.Clear();
        _currentfixations.Add(tempCurrentFixation);
        //Vector3 tempCurrentHeadPosition = _headPositions[1];
        //_headPositions.Clear();
        //_headPositions.Add(tempCurrentHeadPosition);
    }
}
