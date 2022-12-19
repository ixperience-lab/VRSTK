using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using ViveSR.anipal.Eye;
using ViveSR.anipal;
using ViveSR;
using System.Runtime.InteropServices;

public class EyeTrackingAndCalibration : MonoBehaviour
{   
    public List<Vector3> _saccads;
    public List<Vector3> _fixations;
    public List<Vector3> _currentfixations;
    
    private float _currentTime, _lastTime = 0f;

    public int _saccadeCounter = 0;
    public int _fixationCounter = 0;
    public float _saccadeVelocityThreshold = 300f;
    public bool _isEyeDataCallbackFunctionRegistered = false;
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

    //private static long _currentTime, _lastTime = 0;
    private static EyeData_v2 _eyeDataV2 = new EyeData_v2();


    //public static int _countCallbackStatic = 0;
    public static int _sRanipalTimeStamp;
    public static int _sRanipalFrameSequence;
    public static UInt64 _leftEyeDataValidataBitMask, _rightEyeDataValidataBitMask;                                                // The bits explaining the validity of eye data.
    public static float _leftEyeOpennessStatic, _rightEyeOpennessStatic;                                                                       // The level of eye openness.
    public static float _leftPupilDiameterStatic, _rightPupilDiameterStatic;                                   
    public static Vector2 _leftPupilPositionInSensorAreaStatic, _rightPupilPositionInSensorAreaStatic; 
    public static Vector3 _leftGazeOrigin_mm, _rightGazeOrigin_mm;                                                                 // Position of gaze origin.
    public static Vector3 _leftGazeDirectionNormalizedStatic, _rightGazeDirectionNormalizedStatic;                                             // Direction of gaze ray.
    public static Vector3 _leftGazeDirectionNormalizedTranslatedToWorldSpaceStatic, _rightGazeDirectionNormalizedTranslatedToWorldSpaceStatic; 
    public static Vector3 _combinedGazeDirectionNormalizedStatic, _combinedGazeDirectionNormalizedTranslatedToWorldSpaceStatic;
    public static float _leftEyeExpressionData_EyeFrown, _rightEyeExpressionData_EyeFrown;                                         // The level of user's frown.
    public static float _leftEyeExpressionData_EyeSqueeze, _rightEyeExpressionData_EyeSqueeze;                                     // The level to show how the eye is closed tightly.
    public static float _leftEyeExpressionData_EyeWide, _rightEyeExpressionData_EyeWide;                                           // The level to show how the eye is open widely.
    public static double _gazeSensitiveFactor;                                                                                     // The sensitive factor of gaze ray.
    public static float _convergenceDistance_mm;                                                                                   // Distance from the central point of right and left eyes.
    public static bool _convergenceDistanceValidity;                                                                               // Validity of combined data of right and left eyes.
    private static int _trackingImprovementsCount = 0;
    

    // Start is called before the first frame update
    void Start()
    {
        Invoke("CheckAndInitSRanipal", 0.5f);

        _saccads = new List<Vector3>();
        _fixations = new List<Vector3>();
        _currentfixations = new List<Vector3>();

        //if (!SRanipal_Eye_v2.LaunchEyeCalibration()) Calibration();
        
        if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == true && _isEyeDataCallbackFunctionRegistered == false)
        {
            SRanipal_Eye_v2.WrapperRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye_v2.CallbackBasic)EyeDataCallback));
            _isEyeDataCallbackFunctionRegistered = true;
        }
    }

    // Update is called once per frame
    void Update()
    {
        _currentTime = Time.deltaTime;

        _leftEyeOpenness = _leftEyeOpennessStatic;
        _rightEyeOpenness = _rightEyeOpennessStatic;
        _leftPupilDiameter = _leftPupilDiameterStatic;
        _rightPupilDiameter = _rightPupilDiameterStatic;
        _leftPupilPositionInSensorArea = _leftPupilPositionInSensorAreaStatic;
        _rightPupilPositionInSensorArea = _rightPupilPositionInSensorAreaStatic;
        _leftGazeDirectionNormalized = _leftGazeDirectionNormalizedStatic;
        _rightGazeDirectionNormalized = _rightGazeDirectionNormalizedStatic;
        _combinedGazeDirectionNormalized = _combinedGazeDirectionNormalizedStatic;
        _measuredDifferceTime = _currentTime - _lastTime;


        CalculateWorldSpace(_leftGazeDirectionNormalized, ref _leftGazeDirectionNormalizedTranslatedToWorldSpace);
        CalculateWorldSpace(_rightGazeDirectionNormalized, ref _rightGazeDirectionNormalizedTranslatedToWorldSpace);
        CalculateWorldSpace(_combinedGazeDirectionNormalized, ref _combinedGazeDirectionNormalizedTranslatedToWorldSpace);

        Vector3 v = _combinedGazeDirectionNormalizedTranslatedToWorldSpace;
        CalculateSaccades(v, _measuredDifferceTime);

        _lastTime = _currentTime;

        if (SRanipal_Eye_Framework.Status != SRanipal_Eye_Framework.FrameworkStatus.WORKING) return;

        if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == true && _isEyeDataCallbackFunctionRegistered == false)
        {
            SRanipal_Eye.WrapperRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye_v2.CallbackBasic)EyeDataCallback));
            _isEyeDataCallbackFunctionRegistered = true;
        }
        else if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == false && _isEyeDataCallbackFunctionRegistered == true)
        {
            SRanipal_Eye.WrapperUnRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye_v2.CallbackBasic)EyeDataCallback));
            _isEyeDataCallbackFunctionRegistered = false;
        }
    }

    private void OnDisable()
    {
        Release();
    }

    void OnApplicationQuit()
    {
        Release();
    }

    /// <summary>
    /// Release callback thread when disabled or quit
    /// </summary>
    private void Release()
    {
        if (_isEyeDataCallbackFunctionRegistered == true)
        {
            SRanipal_Eye.WrapperUnRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye_v2.CallbackBasic)EyeDataCallback));
            _isEyeDataCallbackFunctionRegistered = false;
        }
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

    /// <summary>
    /// Required class for IL2CPP scripting backend support
    /// </summary>
    internal class MonoPInvokeCallbackAttribute : System.Attribute
    {
        public MonoPInvokeCallbackAttribute() { }
    }

    /// <summary>
    /// Eye tracking data callback thread.
    /// Reports data at ~120hz
    /// MonoPInvokeCallback attribute required for IL2CPP scripting backend
    /// </summary>
    /// <param name="eye_data">Reference to latest eye_data</param>
    [MonoPInvokeCallback]
    private static void EyeDataCallback(ref EyeData_v2 eyeDataV2)
    {
        try
        {
            EyeParameter eyeParameter = new EyeParameter();
            SRanipal_Eye_API.GetEyeParameter(ref eyeParameter);

            _eyeDataV2 = eyeDataV2;

            _sRanipalTimeStamp = _eyeDataV2.timestamp;

            //_currentTime = _sRanipalTimeStamp;// / 1000;

            _sRanipalFrameSequence = _eyeDataV2.frame_sequence;
            _leftEyeDataValidataBitMask = _eyeDataV2.verbose_data.left.eye_data_validata_bit_mask;
            _rightEyeDataValidataBitMask = _eyeDataV2.verbose_data.right.eye_data_validata_bit_mask;
            _leftEyeOpennessStatic = _eyeDataV2.verbose_data.left.eye_openness;
            _rightEyeOpennessStatic = _eyeDataV2.verbose_data.right.eye_openness;
            _leftPupilDiameterStatic = _eyeDataV2.verbose_data.left.pupil_diameter_mm;
            _rightPupilDiameterStatic = _eyeDataV2.verbose_data.right.pupil_diameter_mm;
            _leftPupilPositionInSensorAreaStatic = _eyeDataV2.verbose_data.left.pupil_position_in_sensor_area;
            _rightPupilPositionInSensorAreaStatic = _eyeDataV2.verbose_data.right.pupil_position_in_sensor_area;
            _leftGazeOrigin_mm = _eyeDataV2.verbose_data.left.gaze_origin_mm;
            _rightGazeOrigin_mm = _eyeDataV2.verbose_data.right.gaze_origin_mm;
            _leftGazeDirectionNormalizedStatic = _eyeDataV2.verbose_data.left.gaze_direction_normalized;
            _rightGazeDirectionNormalizedStatic = _eyeDataV2.verbose_data.right.gaze_direction_normalized;
            _gazeSensitiveFactor = eyeParameter.gaze_ray_parameter.sensitive_factor;
            _leftEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.left.eye_frown;
            _rightEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.right.eye_frown;
            _leftEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.left.eye_squeeze;
            _rightEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.right.eye_squeeze;
            _leftEyeExpressionData_EyeWide = _eyeDataV2.expression_data.left.eye_wide;
            _rightEyeExpressionData_EyeWide = _eyeDataV2.expression_data.right.eye_wide;
            _combinedGazeDirectionNormalizedStatic = _eyeDataV2.verbose_data.combined.eye_data.gaze_direction_normalized;
            _convergenceDistanceValidity = _eyeDataV2.verbose_data.combined.convergence_distance_validity;
            _convergenceDistance_mm = _eyeDataV2.verbose_data.combined.convergence_distance_mm;
            _trackingImprovementsCount = _eyeDataV2.verbose_data.tracking_improvements.count;

            ////--------------------------------------------------------------------------------------------------------------------------
            //_leftGazeDirectionNormalizedTranslatedToWorldSpace = transform.TransformDirection(_leftGazeDirectionNormalized);
            //var eyeDirection2 = new Vector2(_leftGazeDirectionNormalizedTranslatedToWorldSpace.x, _leftGazeDirectionNormalizedTranslatedToWorldSpace.z);
            //var headDirection2 = new Vector2(transform.forward.x, transform.forward.z);
            //_leftGazeDirectionNormalizedTranslatedToWorldSpace = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * _leftGazeDirectionNormalizedTranslatedToWorldSpace;
            ////CalculateWorldSpace(_leftGazeDirectionNormalized, ref _leftGazeDirectionNormalizedTranslatedToWorldSpace);
            ////---------------------------------------------------------------------------------------------------------------------------
            //_rightGazeDirectionNormalizedTranslatedToWorldSpace = transform.TransformDirection(_rightGazeDirectionNormalized);
            //eyeDirection2 = new Vector2(_rightGazeDirectionNormalizedTranslatedToWorldSpace.x, _rightGazeDirectionNormalizedTranslatedToWorldSpace.z);
            //headDirection2 = new Vector2(transform.forward.x, transform.forward.z);
            //_rightGazeDirectionNormalizedTranslatedToWorldSpace = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * _rightGazeDirectionNormalizedTranslatedToWorldSpace;
            ////CalculateWorldSpace(_rightGazeDirectionNormalized, ref _rightGazeDirectionNormalizedTranslatedToWorldSpace);
            ////---------------------------------------------------------------------------------------------------------------------------
            //_combinedGazeDirectionNormalizedTranslatedToWorldSpace = transform.TransformDirection(_combinedGazeDirectionNormalized);
            //eyeDirection2 = new Vector2(_combinedGazeDirectionNormalizedTranslatedToWorldSpace.x, _combinedGazeDirectionNormalizedTranslatedToWorldSpace.z);
            //headDirection2 = new Vector2(transform.forward.x, transform.forward.z);
            //_combinedGazeDirectionNormalizedTranslatedToWorldSpace = Quaternion.AngleAxis(-2 * Vector2.SignedAngle(eyeDirection2, headDirection2), Vector3.up) * _combinedGazeDirectionNormalizedTranslatedToWorldSpace;
            ////CalculateWorldSpace(_combinedGazeDirectionNormalized, ref _combinedGazeDirectionNormalizedTranslatedToWorldSpace);
            ////--------------------------------------------------------------------------------------------------------------------------

            //Vector3 v = _combinedGazeDirectionNormalizedTranslatedToWorldSpace;//CalculateGazeVector(_combinedGazeDirectionNormalizedTranslatedToWorldSpace, transform.position);

            //long measuredTime = _currentTime - _lastTime;

            //if (_currentfixations.Count < 2)
            //    _currentfixations.Add(v);
            //else
            //{
            //    //float visualAngle = CalculateVisualAngle(_currentfixations[0], _currentfixations[1]);
            //    float visualAngle = Mathf.Acos(Vector3.Dot(_currentfixations[0], _currentfixations[1]) / (_currentfixations[0].magnitude * _currentfixations[1].magnitude)); //(gazeVector0.sqrMagnitude * gazeVector01.sqrMagnitude));

            //    if (measuredTime == 0) measuredTime = 1;
            //    float gazeVelocity = visualAngle / measuredTime;//CalculateGazeVelocity(visualAngle, timeDiff);

            //    bool isSaccade = false;

            //    if (gazeVelocity > _saccadeVelocityThreshold)
            //    {
            //        _saccads.Clear();
            //        _saccads.Add(_currentfixations[0]);
            //        _saccads.Add(_currentfixations[1]);
            //        _saccadeCounter++;
            //        isSaccade = true;
            //    }

            //    for (int i = 0; i < _currentfixations.Count; i++)
            //        if (!isSaccade)
            //        {
            //            _fixations.Add(_currentfixations[i]);
            //            _fixationCounter++;
            //        }

            //    Vector3 temp = _currentfixations[1];
            //    _currentfixations.Clear();
            //    _currentfixations.Add(temp);
            //}

            //CalculateSaccades(v, measuredTime);
        }
        catch(Exception e)
        {
            Debug.LogWarning("Catch Exception: " + e.ToString());
        }

        //_lastTime = _currentTime;
        //_countCallback++;
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

    //private Vector3 CalculateGazeVector(Vector3 gazeIntersectionPoint, Vector3 averagedHeadPostion)
    //{
    //    return gazeIntersectionPoint - averagedHeadPostion;
    //}

    //private float CalculateVisualAngle(Vector3 gazeVector0, Vector3 gazeVector01)
    //{
    //    return Mathf.Acos(Vector3.Dot(gazeVector0, gazeVector01) / (gazeVector0.magnitude * gazeVector01.magnitude)); //(gazeVector0.sqrMagnitude * gazeVector01.sqrMagnitude));
    //}

    //private float CalculateGazeVelocity(float visualAngle, long timeDiff)
    //{
    //    if (timeDiff == 0) timeDiff = 1;
    //    return visualAngle / timeDiff;
    //}

    private void CalculateSaccades(Vector3 gazeVector, float timeDiff)
    {
        if (_currentfixations.Count < 2)
        {
            _currentfixations.Add(gazeVector);
            return;
        }

        //float visualAngle = CalculateVisualAngle(_currentfixations[0], _currentfixations[1]);
        float visualAngle = Mathf.Acos(Vector3.Dot(_currentfixations[0], _currentfixations[1]) / (_currentfixations[0].magnitude * _currentfixations[1].magnitude)); //(gazeVector0.sqrMagnitude * gazeVector01.sqrMagnitude));

        if (timeDiff == 0) timeDiff = 1f;
        float gazeVelocity = visualAngle / timeDiff;//CalculateGazeVelocity(visualAngle, timeDiff);

        _measuredVisualAngle = visualAngle;
        _measuredVelocity = gazeVelocity;

        bool isSaccade = false;

        if (gazeVelocity > _saccadeVelocityThreshold)
        {
            _saccads.Clear();
            _fixations.Clear();
            _saccads.Add(_currentfixations[0]);
            _saccads.Add(_currentfixations[1]);
            _saccadeCounter++;
            isSaccade = true;
        }

        for (int i = 0; i < _currentfixations.Count; i++)
            if (!isSaccade)
            {
                _fixations.Add(_currentfixations[i]);
                _fixationCounter++;
            }

        Vector3 temp = _currentfixations[1];
        _currentfixations.Clear();
        _currentfixations.Add(temp);
    }
}
