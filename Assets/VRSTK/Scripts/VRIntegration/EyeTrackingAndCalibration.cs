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
    private float _angle = 8;                                                               // Angle (degrees) between center target and L/R target.        Recommended: 8-10°
    private long _saccadeEndTime = 0;
    private long _measureTime, _currentTime, measureEndTime = 0;
    private EyeData_v2 _eyeDataV2 = new EyeData_v2();
    
    
    public int _saccadeCounter = 0, _endBuffer = 3, _saccadeTimer = 30;
    public int _fixationCounter = 0;
    float _timeout = 1.0f, _initialTimer = 0.0f;

    public int _countCallback = 0;
    public float _sRanipalTimeStamp;
    public int _sRanipalFrameSequence;
    public bool _isEyeDataCallbackFunctionRegistered = false;
    public UInt64 _leftEyeDataValidataBitMask, _rightEyeDataValidataBitMask;                // The bits explaining the validity of eye data.
    public float _leftEyeOpenness, _rightEyeOpenness;                                       // The level of eye openness.
    public float _leftPupilDiameter, _rightPupilDiameter;                                   
    public Vector2 _leftPupilPositionInSensorArea, _rightPupilPositionInSensorArea;
    public Vector3 _leftGazeOrigin_mm, _rightGazeOrigin_mm;                                 // Position of gaze origin.
    public Vector3 _leftGazeDirectionNormalized, _rightGazeDirectionNormalized;             // Direction of gaze ray.
    public float _leftEyeExpressionData_EyeFrown, _rightEyeExpressionData_EyeFrown;         // The level of user's frown.
    public float _leftEyeExpressionData_EyeSqueeze, _rightEyeExpressionData_EyeSqueeze;     // The level to show how the eye is closed tightly.
    public float _leftEyeExpressionData_EyeWide, _rightEyeExpressionData_EyeWide;           // The level to show how the eye is open widely.
    public double _gazeSensitiveFactor;                                                     // The sensitive factor of gaze ray.
    public float _convergenceDistance_mm;                                                   // Distance from the central point of right and left eyes.
    public bool _convergenceDistanceValidity;                                               // Validity of combined data of right and left eyes.
    public bool _isUserNeedCalibration;                                                     
    public bool _calibrationResult;                                                         
    private int _trackingImprovementsCount = 0;
    

    // Start is called before the first frame update
    void Start()
    {
        Invoke("CheckAndInitSRanipal", 0.5f);
        
        if (!SRanipal_Eye_v2.LaunchEyeCalibration()) Calibration();
        
        if (SRanipal_Eye_Framework.Instance.EnableEyeDataCallback == true && _isEyeDataCallbackFunctionRegistered == false)
        {
            SRanipal_Eye_v2.WrapperRegisterEyeDataCallback(Marshal.GetFunctionPointerForDelegate((SRanipal_Eye_v2.CallbackBasic)EyeDataCallback));
            _isEyeDataCallbackFunctionRegistered = true;
        }
    }

    // Update is called once per frame
    void Update()
    {
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
    private void EyeDataCallback(ref EyeData_v2 eyeDataV2)
    {
        EyeParameter eyeParameter = new EyeParameter();
        SRanipal_Eye_API.GetEyeParameter(ref eyeParameter);
        
        try
        {
            _eyeDataV2 = eyeDataV2;

            _sRanipalTimeStamp = _eyeDataV2.timestamp;
            _sRanipalFrameSequence = _eyeDataV2.frame_sequence;
            _leftEyeDataValidataBitMask = _eyeDataV2.verbose_data.left.eye_data_validata_bit_mask;
            _rightEyeDataValidataBitMask = _eyeDataV2.verbose_data.right.eye_data_validata_bit_mask;
            _leftEyeOpenness = _eyeDataV2.verbose_data.left.eye_openness;
            _rightEyeOpenness = _eyeDataV2.verbose_data.right.eye_openness;
            _leftPupilDiameter = _eyeDataV2.verbose_data.left.pupil_diameter_mm;
            _rightPupilDiameter = _eyeDataV2.verbose_data.right.pupil_diameter_mm;
            _leftPupilPositionInSensorArea = _eyeDataV2.verbose_data.left.pupil_position_in_sensor_area;
            _rightPupilPositionInSensorArea = _eyeDataV2.verbose_data.right.pupil_position_in_sensor_area;
            _leftGazeOrigin_mm = _eyeDataV2.verbose_data.left.gaze_origin_mm;
            _rightGazeOrigin_mm = _eyeDataV2.verbose_data.right.gaze_origin_mm;
            _leftGazeDirectionNormalized = _eyeDataV2.verbose_data.left.gaze_direction_normalized;
            _rightGazeDirectionNormalized = _eyeDataV2.verbose_data.right.gaze_direction_normalized;
            _gazeSensitiveFactor = eyeParameter.gaze_ray_parameter.sensitive_factor;
            _leftEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.left.eye_frown;
            _rightEyeExpressionData_EyeFrown = _eyeDataV2.expression_data.right.eye_frown;
            _leftEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.left.eye_squeeze;
            _rightEyeExpressionData_EyeSqueeze = _eyeDataV2.expression_data.right.eye_squeeze;
            _leftEyeExpressionData_EyeWide = _eyeDataV2.expression_data.left.eye_wide;
            _rightEyeExpressionData_EyeWide = _eyeDataV2.expression_data.right.eye_wide;
            _convergenceDistanceValidity = _eyeDataV2.verbose_data.combined.convergence_distance_validity;
            _convergenceDistance_mm = _eyeDataV2.verbose_data.combined.convergence_distance_mm;
            _trackingImprovementsCount = _eyeDataV2.verbose_data.tracking_improvements.count;

            _countCallback++;
        }
        catch(Exception e)
        {
            Debug.LogWarning("Catch Exception: " + e.ToString());
        }
    }
}
