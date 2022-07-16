using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EmotivUnityPlugin;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            public class CortexBrainComputerInterface : MonoBehaviour
            {
                private enum DataStreamType { Motion, EEG, PerformanceMetrics, BandPowerData, DevelopmentInformation, SystemInformation };
                [SerializeField]
                public EmotivParameter _isMotionTracking = new EmotivParameter();
                private bool _previousIsMotionTrackingValue;

                [SerializeField]
                public EmotivParameter _isEegTrackingActive = new EmotivParameter();
                private bool _previousIsEegTrackingActiveValue;

                [SerializeField]
                public EmotivParameter _isPerformanceMetricsTrackingActive = new EmotivParameter();
                private bool _previousIsPerformanceMetricsTrackingActiveValue;

                [SerializeField]
                public EmotivParameter _isBandPowerTrackingActive = new EmotivParameter();
                private bool _previousIsBandPowerTrackingActiveValue;


                private EmotivUnityItf _emotivUnityItf = EmotivUnityItf.Instance;
                private float _timerDataUpdate = 0;
                private const float TIME_UPDATE_DATA = 1f;
                public bool _isDataBufferUsing = false; // default subscribed data will not saved to Data buffer

                public void Awake()
                {
                    _isMotionTracking.name = "isMotionTracking";
                    _isMotionTracking.hideFromInspector = true;
                    _isMotionTracking.value = false;
                    _previousIsMotionTrackingValue = false;

                    _isEegTrackingActive.name = "isEegTrackingActive";
                    _isEegTrackingActive.hideFromInspector = true;
                    _isEegTrackingActive.value = false;
                    _previousIsEegTrackingActiveValue = false;

                    _isPerformanceMetricsTrackingActive.name = "isPerformanceMetricsTrackingActive";
                    _isPerformanceMetricsTrackingActive.hideFromInspector = true;
                    _isPerformanceMetricsTrackingActive.value = false;
                    _previousIsPerformanceMetricsTrackingActiveValue = false;

                    _isBandPowerTrackingActive.name = "isBandPowerTrackingActive";
                    _isBandPowerTrackingActive.hideFromInspector = true;
                    _isBandPowerTrackingActive.value = false;
                    _previousIsBandPowerTrackingActiveValue = false;
                }

                // Start is called before the first frame update
                void Start()
                {
                    // init EmotivUnityItf without data buffer using
                    _emotivUnityItf.Init(CortexAppConfig.ClientId, CortexAppConfig.ClientSecret, CortexAppConfig.AppName, CortexAppConfig.AppVersion, _isDataBufferUsing);

                    Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                    // Start
                    _emotivUnityItf.Start();

                    //_dataStream.SessionActivatedOK += SessionActivatedOK;
                    //_dataStream.LicenseValidTo += OnLicenseValidTo;
                }

                // Update is called once per frame
                void Update()
                {
                    _timerDataUpdate += Time.deltaTime;
                    if (_timerDataUpdate < TIME_UPDATE_DATA)
                        return;

                    _timerDataUpdate -= TIME_UPDATE_DATA;
                    
                    if (!_emotivUnityItf.IsAuthorizedOK)
                    {
                        return; 
                    }

                    if (!_emotivUnityItf.IsSessionCreated)
                    {
                        //_emotivUnityItf.CreateSessionWithHeadset(HeadsetId.text);
                        return;
                    }

                    if (_emotivUnityItf.IsSessionCreated)
                    {

                        if (_isMotionTracking.value != _previousIsMotionTrackingValue)
                        {
                            _previousIsMotionTrackingValue = _isMotionTracking.value;
                            if (_isMotionTracking.value == true)
                            {
                                _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                                _emotivUnityItf.SubscribeData(GetStreamsList(DataStreamType.Motion));
                            }
                            else
                            {
                                _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                                _emotivUnityItf.UnSubscribeData(GetStreamsList(DataStreamType.Motion));
                            }
                        }
                    }

                    if (_isPerformanceMetricsTrackingActive.value != _previousIsPerformanceMetricsTrackingActiveValue)
                    {
                        _previousIsPerformanceMetricsTrackingActiveValue = _isPerformanceMetricsTrackingActive.value;

                        if (_isPerformanceMetricsTrackingActive.value == true)
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.SubscribeData(GetStreamsList(DataStreamType.PerformanceMetrics));
                        }
                        else
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.UnSubscribeData(GetStreamsList(DataStreamType.PerformanceMetrics));
                        }
                    }

                    if (_isEegTrackingActive.value != _previousIsEegTrackingActiveValue)
                    {
                        _previousIsEegTrackingActiveValue = _isEegTrackingActive.value;

                        if (_isEegTrackingActive.value == true)
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.SubscribeData(GetStreamsList(DataStreamType.EEG));
                        }
                        else
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.UnSubscribeData(GetStreamsList(DataStreamType.EEG));
                        }
                    }

                    if (_isBandPowerTrackingActive.value != _previousIsBandPowerTrackingActiveValue)
                    {
                        _previousIsBandPowerTrackingActiveValue = _isBandPowerTrackingActive.value;

                        if (_isBandPowerTrackingActive.value == true)
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.SubscribeData(GetStreamsList(DataStreamType.BandPowerData));
                        }
                        else
                        {
                            _emotivUnityItf.DataSubLog = ""; // clear data subscribing log
                            _emotivUnityItf.UnSubscribeData(GetStreamsList(DataStreamType.BandPowerData));
                        }
                    }
                }

                void OnApplicationQuit()
                {
                    Debug.Log("Application ending after " + Time.time + " seconds");
                    _emotivUnityItf.Stop();
                }

                private List<string> GetStreamsList(DataStreamType datataStreamType)
                {
                    List<string> _streams = new List<string> { };
                    if (datataStreamType == DataStreamType.EEG)
                    {
                        _streams.Add("eeg");
                    }
                    if (datataStreamType == DataStreamType.Motion)
                    {
                        _streams.Add("mot");
                    }
                    if (datataStreamType == DataStreamType.PerformanceMetrics)
                    {
                        _streams.Add("met");
                    }
                    if (datataStreamType == DataStreamType.DevelopmentInformation)
                    {
                        _streams.Add("dev");
                    }
                    if (datataStreamType == DataStreamType.SystemInformation)
                    {
                        _streams.Add("sys");
                    }
                    //if (EQToggle.isOn)
                    //{
                    //    _streams.Add("eq");
                    //}
                    if (datataStreamType == DataStreamType.BandPowerData)
                    {
                        _streams.Add("pow");
                    }
                    //if (FEToggle.isOn)
                    //{
                    //    _streams.Add("fac");
                    //}
                    //if (COMToggle.isOn)
                    //{
                    //    _streams.Add("com");
                    //}
                    return _streams;
                }
            }
        }
    }
}
