using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EmotivUnityPlugin;
using Newtonsoft.Json.Linq;
using System.IO;
using VRSTK.Scripts.TestControl;
using System;

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

                [SerializeField]
                public EmotivParameter _isDevDataTrackingActive = new EmotivParameter();
                private bool _previousIsDevDataTrackingActiveValue;

                [SerializeField]
                public EmotivParameter _isSysEventDataTrackingActive = new EmotivParameter();
                private bool _previousIsSysEventDataTrackingActiveValue;

                [SerializeField]
                private string _motionDataMessage;

                public string MotionDataMessage
                {
                    get { return _motionDataMessage; }
                    set { _motionDataMessage = value; }
                }

                [SerializeField]
                private string _rawEegDataMessage;

                public string RawEegDataMessage
                {
                    get { return _rawEegDataMessage; }
                    set { _rawEegDataMessage = value; }
                }

                [SerializeField]
                private string _bandPowerDataMessage;

                public string BandPowerDataMessage
                {
                    get { return _bandPowerDataMessage; }
                    set { _bandPowerDataMessage = value; }
                }

                [SerializeField]
                private string _performanceMetricsDataMessage;

                public string PerformanceMetricsDataMessage
                {
                    get { return _performanceMetricsDataMessage; }
                    set { _performanceMetricsDataMessage = value; }
                }

                [SerializeField]
                private string _devDataMessage;

                public string DevDataMessage
                {
                    get { return _devDataMessage; }
                    set { _devDataMessage = value; }
                }

                [SerializeField]
                private string _sysEventDataMessage;

                public string SysEventDataMessage
                {
                    get { return _sysEventDataMessage; }
                    set { _sysEventDataMessage = value; }
                }

                [SerializeField]
                public string _path;

                [SerializeField]
                public string _rawEEGFileName;

                private string _privateRawEEGFileName = "EmotivRawDataCortexEEG";

                public GameObject _testController;

                //DataStreamManager _dataStream = DataStreamManager.Instance;
                //Logger _logger = Logger.Instance;
                //private HeadsetFinder _headsetFinder = HeadsetFinder.Instance;

                DataStreamManager _dataStream = null;
                Logger _logger = null;
                private HeadsetFinder _headsetFinder = null;

                private Headset _headsetInformation;

                bool _isAuthorizedOK = false;
                bool _isSessionCreated = false;
                
                bool _isQueryHeadset = false;
                bool _isQueryHeadsetDeactivated = false;
                bool _isConnectedToDevice = false;
                
                float _timerCounter_queryHeadset = 0;
                const float TIME_QUERY_HEADSET = 2.0f;

                private string _workingHeadsetId = "";

                private float _timerDataUpdate = 0;
                private const float TIME_UPDATE_DATA = 1f;
                public bool _isDataBufferUsing = false; // default subscribed data will not saved to Data buffer

                public void Awake()
                {
                    _isMotionTracking.name = "isMotionTracking";
                    _isMotionTracking.hideFromInspector = true;
                    _isMotionTracking.value = true;
                    _previousIsMotionTrackingValue = false;

                    _isEegTrackingActive.name = "isEegTrackingActive";
                    _isEegTrackingActive.hideFromInspector = true;
                    _isEegTrackingActive.value = true;
                    _previousIsEegTrackingActiveValue = false;

                    _isPerformanceMetricsTrackingActive.name = "isPerformanceMetricsTrackingActive";
                    _isPerformanceMetricsTrackingActive.hideFromInspector = true;
                    _isPerformanceMetricsTrackingActive.value = true;
                    _previousIsPerformanceMetricsTrackingActiveValue = false;

                    _isBandPowerTrackingActive.name = "isBandPowerTrackingActive";
                    _isBandPowerTrackingActive.hideFromInspector = true;
                    _isBandPowerTrackingActive.value = true;
                    _previousIsBandPowerTrackingActiveValue = false;


                    _isDevDataTrackingActive.name = "isDevDataTrackingActive";
                    _isDevDataTrackingActive.hideFromInspector = true;
                    _isDevDataTrackingActive.value = true;
                    _previousIsDevDataTrackingActiveValue = false;

                
                    _isSysEventDataTrackingActive.name = "isSysEventDataTrackingActive";
                    _isSysEventDataTrackingActive.hideFromInspector = true;
                    _isSysEventDataTrackingActive.value = true;
                    _previousIsSysEventDataTrackingActiveValue = false;
                }

                // Start is called before the first frame update
                void Start()
                {
                    if (!_rawEEGFileName.Equals(_privateRawEEGFileName))
                    {
                        _rawEEGFileName = _privateRawEEGFileName;
                        //create for every stage and id a new file
                        if (_testController != null)
                        {
                            TestController testController = _testController.GetComponent<TestController>();
                            for (int i = 0; i < testController.testStages.Length; i++)
                            {
                                if (testController.testStages[i].active)
                                {
                                    _rawEEGFileName += "_" + testController.testStages[i].name;
                                    TestStage testStage = testController.testStages[i].GetComponent<TestStage>();

                                    for (int j = 0; j < testStage.startProperties.Length; j++)
                                    {
                                        if (testStage.startProperties[j].text.text.ToLower().Contains("id"))
                                            _rawEEGFileName += "_" + testStage.startProperties[j].GetValue();
                                        if (testStage.startProperties[j].text.text.ToLower().Contains("condition") && testStage.startProperties[j].GetValue().ToLower().Equals("true"))
                                            _rawEEGFileName += "_" + testStage.startProperties[j].text.text;
                                    }

                                    _rawEEGFileName += "_" + DateTime.Now.ToString("yyyy-MM-dd_hh-mm-ss") + ".txt";
                                    break;
                                }
                            }
                        }

                        if (!_rawEEGFileName.Contains(".txt"))
                            _rawEEGFileName += ".txt";
                    }

                    if (!_isAuthorizedOK)
                    {
                        if (_dataStream != null && _logger != null && _headsetFinder == null)
                        {
                            // init EmotivUnityItf without data buffer using
                            Init(_isDataBufferUsing);

                            // Init logger
                            _logger.Init();


                            Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                            // Start
                            _dataStream.StartAuthorize();
                        }
                        else
                        {
                            _dataStream = DataStreamManager.Instance;
                            _logger = Logger.Instance;
                            _headsetFinder = HeadsetFinder.Instance;
                            
                            // init EmotivUnityItf without data buffer using
                            Init(_isDataBufferUsing);

                            // Init logger
                            _logger.Init();

                            Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                            // Start
                            _dataStream.StartAuthorize();
                        }
                    }
                }

                // Update is called once per frame
                void Update()
                {
                    if (_isQueryHeadset)
                    {
                        Debug.Log("=============== Count GetDetectedHeadsets : " + _dataStream.GetDetectedHeadsets().Count);
                        if (_dataStream.GetDetectedHeadsets().Count > 0 && !_isConnectedToDevice)
                        {
                            Debug.Log("=============== Count GetDetectedHeadsets : " + _dataStream.GetDetectedHeadsets().Count);
                            GetHeatsetInformationOfIndexZero();
                            _isConnectedToDevice = true;
                        }

                        _timerCounter_queryHeadset += Time.deltaTime;
                        if (_timerCounter_queryHeadset > TIME_QUERY_HEADSET && !_isSessionCreated)
                        {
                            Debug.Log("=============== time for query Headset ");
                            _timerCounter_queryHeadset -= TIME_QUERY_HEADSET;
                            _isQueryHeadset = false;
                        }
                    }

                    if (!_isQueryHeadset)
                    {
                        Debug.Log("=============== QueryHeadsets");
                        _dataStream.QueryHeadsets();
                        _isQueryHeadsetDeactivated = false;
                        _isQueryHeadset = true;
                    }

                    if (!_isAuthorizedOK)
                    {
                        return; 
                    }

                    if (!_isSessionCreated)
                    {
                        if (_headsetInformation == null)
                            return;
                        
                        CreateSessionWithHeadset(_headsetInformation.HeadsetID);
                        return;
                    }

                    if (_isSessionCreated)
                    {
                        if (_isMotionTracking.value != _previousIsMotionTrackingValue)
                        {
                            _previousIsMotionTrackingValue = _isMotionTracking.value;
                            if (_isMotionTracking.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.Motion));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.Motion));
                        }

                        if (_isPerformanceMetricsTrackingActive.value != _previousIsPerformanceMetricsTrackingActiveValue)
                        {
                            _previousIsPerformanceMetricsTrackingActiveValue = _isPerformanceMetricsTrackingActive.value;

                            if (_isPerformanceMetricsTrackingActive.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.PerformanceMetrics));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.PerformanceMetrics));
                        }

                        if (_isEegTrackingActive.value != _previousIsEegTrackingActiveValue)
                        {
                            _previousIsEegTrackingActiveValue = _isEegTrackingActive.value;

                            if (_isEegTrackingActive.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.EEG));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.EEG));
                        }

                        if (_isBandPowerTrackingActive.value != _previousIsBandPowerTrackingActiveValue)
                        {
                            _previousIsBandPowerTrackingActiveValue = _isBandPowerTrackingActive.value;

                            if (_isBandPowerTrackingActive.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.BandPowerData));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.BandPowerData));
                        }

                        if (_isDevDataTrackingActive.value != _previousIsDevDataTrackingActiveValue)
                        {
                            _previousIsDevDataTrackingActiveValue = _isDevDataTrackingActive.value;

                            if (_isDevDataTrackingActive.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.DevelopmentInformation));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.DevelopmentInformation));
                        }

                        if (_isSysEventDataTrackingActive.value != _previousIsSysEventDataTrackingActiveValue)
                        {
                            _previousIsSysEventDataTrackingActiveValue = _isSysEventDataTrackingActive.value;

                            if (_isSysEventDataTrackingActive.value == true)
                                _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.SystemInformation));
                            else
                                _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.SystemInformation));
                        }
                    }

                    _timerDataUpdate += Time.deltaTime;
                    if (_timerDataUpdate < TIME_UPDATE_DATA)
                        return;

                    _timerDataUpdate -= TIME_UPDATE_DATA;

                }

                void OnApplicationQuit()
                {
                    if (_dataStream != null)
                    {
                        _isMotionTracking.value = false;
                        _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.Motion));
                        _previousIsMotionTrackingValue = _isMotionTracking.value;

                        _isEegTrackingActive.value = false;
                        _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.EEG));
                        _previousIsEegTrackingActiveValue = _isEegTrackingActive.value;

                        _isPerformanceMetricsTrackingActive.value = false;
                        _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.PerformanceMetrics));
                        _previousIsPerformanceMetricsTrackingActiveValue = _isPerformanceMetricsTrackingActive.value;

                        _isBandPowerTrackingActive.value = false;
                        _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.BandPowerData));
                        _previousIsBandPowerTrackingActiveValue = _isBandPowerTrackingActive.value;

                        _isDevDataTrackingActive.value = false;
                        _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.DevelopmentInformation));
                        _previousIsDevDataTrackingActiveValue = _isDevDataTrackingActive.value;

                        _isSysEventDataTrackingActive.value = false;
                        _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.SystemInformation));
                        _previousIsSysEventDataTrackingActiveValue = _isSysEventDataTrackingActive.value;
                    }

                    Debug.Log("Application ending after " + Time.time + " seconds");
                    Stop();

                    if (_dataStream != null)
                        _dataStream = null;
                    if (_logger != null)
                        _logger = null;
                    if (_headsetFinder != null)
                        _headsetFinder = null;
                }

                void OnEnable()
                {
                    if (!_rawEEGFileName.Equals(_privateRawEEGFileName))
                    {
                        _rawEEGFileName = _privateRawEEGFileName;
                        //create for every stage and id a new file
                        if (_testController != null)
                        {
                            TestController testController = _testController.GetComponent<TestController>();
                            for (int i = 0; i < testController.testStages.Length; i++)
                            {
                                if (testController.testStages[i].active)
                                {
                                    _rawEEGFileName += "_" + testController.testStages[i].name;
                                    TestStage testStage = testController.testStages[i].GetComponent<TestStage>();

                                    for (int j = 0; j < testStage.startProperties.Length; j++)
                                    {
                                        if (testStage.startProperties[j].text.text.ToLower().Contains("id"))
                                            _rawEEGFileName += "_" + testStage.startProperties[j].GetValue();
                                        if (testStage.startProperties[j].text.text.ToLower().Contains("condition") && testStage.startProperties[j].GetValue().ToLower().Equals("true"))
                                            _rawEEGFileName += "_" + testStage.startProperties[j].text.text;
                                    }

                                    _rawEEGFileName += "_" + DateTime.Now.ToString("yyyy-MM-dd_hh-mm-ss") + ".txt";
                                    break;
                                }
                            }
                        }

                        if (!_rawEEGFileName.Contains(".txt"))
                            _rawEEGFileName += ".txt";
                    }

                    _isMotionTracking.value = true;
                    _isEegTrackingActive.value = true;
                    _isPerformanceMetricsTrackingActive.value = true;
                    _isBandPowerTrackingActive.value = true;
                    _isDevDataTrackingActive.value = true;
                    _isSysEventDataTrackingActive.value = true;

                    if (!_isAuthorizedOK)
                    {
                        if (_dataStream != null && _logger != null && _headsetFinder == null)
                        {
                            // init EmotivUnityItf without data buffer using
                            Init(_isDataBufferUsing);

                            // Init logger
                            _logger.Init();


                            Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                            // Start
                            _dataStream.StartAuthorize();
                        }
                        else
                        {
                            _dataStream = DataStreamManager.Instance;
                            _logger = Logger.Instance;
                            _headsetFinder = HeadsetFinder.Instance;

                            // init EmotivUnityItf without data buffer using
                            Init(_isDataBufferUsing);

                            // Init logger
                            _logger.Init();

                            Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                            // Start
                            _dataStream.StartAuthorize();
                        }
                    }
                }

                void OnDisable()
                {
                    _isMotionTracking.value = false;
                    _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.Motion));
                    _previousIsMotionTrackingValue = _isMotionTracking.value;

                    _isEegTrackingActive.value = false;
                    _dataStream.SubscribeMoreData(GetStreamsList(DataStreamType.EEG));
                    _previousIsEegTrackingActiveValue = _isEegTrackingActive.value;

                    _isPerformanceMetricsTrackingActive.value = false;
                    _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.PerformanceMetrics));
                    _previousIsPerformanceMetricsTrackingActiveValue = _isPerformanceMetricsTrackingActive.value;

                    _isBandPowerTrackingActive.value = false;
                    _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.BandPowerData));
                    _previousIsBandPowerTrackingActiveValue = _isBandPowerTrackingActive.value;

                    _isDevDataTrackingActive.value = false;
                    _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.DevelopmentInformation));
                    _previousIsDevDataTrackingActiveValue = _isDevDataTrackingActive.value;

                    _isSysEventDataTrackingActive.value = false;
                    _dataStream.UnSubscribeData(GetStreamsList(DataStreamType.SystemInformation));
                    _previousIsSysEventDataTrackingActiveValue = _isSysEventDataTrackingActive.value;

                    Debug.Log("Object deactivated after " + Time.time + " seconds");
                    Stop();

                    if (_dataStream != null)
                        _dataStream = null;
                    if (_logger != null)
                        _logger = null;
                    if (_headsetFinder != null)
                        _headsetFinder = null;
                }

                // Init
                public void Init(bool isDataBufferUsing = false)
                {
                    if (string.IsNullOrEmpty(CortexAppConfig.ClientId) || string.IsNullOrEmpty(CortexAppConfig.ClientSecret))
                    {
                        UnityEngine.Debug.LogError("The clientId or clientSecret is empty. Please fill them before starting.");
                        return;
                    }
                    _dataStream.SetAppConfig(CortexAppConfig.ClientId, CortexAppConfig.ClientSecret, CortexAppConfig.AppVersion, CortexAppConfig.AppName, CortexAppConfig.TmpAppDataDir, CortexAppConfig.AppUrl, EmotivAppslicationPath());
                    _dataStream.IsDataBufferUsing = isDataBufferUsing;
                   
                    // binding
                    _dataStream.LicenseValidTo += OnLicenseValidTo;
                    _dataStream.SessionActivatedOK += OnSessionActiveOK;

                    // if do not use data buffer to store data, we need to handle data stream signal
                    if (!isDataBufferUsing)
                    {
                        _dataStream.EEGDataReceived += OnEEGDataReceived;
                        _dataStream.MotionDataReceived += OnMotionDataReceived;
                        _dataStream.DevDataReceived += OnDevDataReceived;
                        _dataStream.PerfDataReceived += OnPerfDataReceived;
                        _dataStream.BandPowerDataReceived += OnBandPowerDataReceived;
                        _dataStream.InformSuccessSubscribedData += OnInformSuccessSubscribedData;

                    }

                    _dataStream.FacialExpReceived += OnFacialExpReceived;
                    //_dataStream.MentalCommandReceived += OnMentalCommandReceived;
                    _dataStream.SysEventsReceived += OnSysEventsReceived;
                }

                /// <summary>
                /// Stop program to clear data, stop queryHeadset
                /// </summary>
                public void Stop()
                {
                    if (_dataStream != null)
                        _dataStream.Stop();
                    _isAuthorizedOK = false;
                    //_isProfileLoaded = false;
                    _isSessionCreated = false;
                    _workingHeadsetId = "";

                    System.Threading.Thread.Sleep(5000);
                }

                private void GetHeatsetInformationOfIndexZero()
                {
                    _headsetInformation = _dataStream.GetDetectedHeadsets()[0];
                }

                public void CreateSessionWithHeadset(string headsetId)
                {
                    // start data stream without streams -> create session with the headset
                    if (_isAuthorizedOK)
                        _dataStream.StartDataStream(new List<string>(), headsetId);
                    else
                        UnityEngine.Debug.LogWarning("Please wait authorize successfully before creating session with headset " + headsetId);
                }

                // Event handlers
                private void OnLicenseValidTo(object sender, System.DateTime validTo)
                {
                    UnityEngine.Debug.Log("OnLicenseValidTo: the license valid to " + Utils.ISODateTimeToString(validTo));
                    _isAuthorizedOK = true;
//                    _messageLog = "Authorizing process done.";
                }

                private void OnSessionActiveOK(object sender, string headsetId)
                {
                    _isSessionCreated = true;
                    _workingHeadsetId = headsetId;

                    // activating tracking parameters
                    {
                        _isMotionTracking.hideFromInspector = false;
                        _isEegTrackingActive.hideFromInspector = false;
                        _isPerformanceMetricsTrackingActive.hideFromInspector = false;
                        _isBandPowerTrackingActive.hideFromInspector = false;
                        _isDevDataTrackingActive.hideFromInspector = true;
                        _isSysEventDataTrackingActive.hideFromInspector = true;
                    }
                    UnityEngine.Debug.Log("A session working with " + headsetId + " is activated successfully.");
                    // _messageLog = "A session working with " + headsetId + " is activated successfully.";
                }

                private void OnInformLoadUnLoadProfileDone(object sender, bool isProfileLoaded)
                {
//                    _isProfileLoaded = isProfileLoaded;
//                    if (isProfileLoaded)
                    //{
                    //    _messageLog = "The profile is loaded successfully.";
                    //}
                    //else
                    //{
                    //    _messageLog = "The profile is unloaded successfully.";
                    //}
                }

                private void OnInformStartRecordResult(object sender, Record record)
                {
                    UnityEngine.Debug.Log("OnInformStartRecordResult recordId: " + record.Uuid + ", title: "
                        + record.Title + ", startDateTime: " + record.StartDateTime);
                    //_isRecording = true;
                    //_messageLog = "The record " + record.Title + " is created at " + record.StartDateTime;

                }

                private void OnInformStopRecordResult(object sender, Record record)
                {
                    UnityEngine.Debug.Log("OnInformStopRecordResult recordId: " + record.Uuid + ", title: "
                        + record.Title + ", startDateTime: " + record.StartDateTime + ", endDateTime: " + record.EndDateTime);
//                    _isRecording = false;
//                    _messageLog = "The record " + record.Title + " is ended at " + record.EndDateTime;

                }

                private void OnInformMarkerResult(object sender, JObject markerObj)
                {
                    //UnityEngine.Debug.Log("OnInformMarkerResult");
                    //_messageLog = "The marker " + markerObj["uuid"].ToString() + ", label: "
                    //    + markerObj["label"].ToString() + ", value: " + markerObj["value"].ToString()
                    //    + ", type: " + markerObj["type"].ToString() + ", started at: " + markerObj["startDatetime"].ToString();
                }

                private void OnInformSuccessSubscribedData(object sender, List<string> successStreams)
                {
                    string tmpText = "The streams: ";
                    foreach (var item in successStreams)
                    {
                        tmpText = tmpText + item + "; ";
                    }
                    tmpText = tmpText + " are subscribed successfully. The output data will be shown on the console log.";
                    UnityEngine.Debug.Log(tmpText);
                    //_messageLog = tmpText;
                }

                // Handle events  if we do not use data buffer of Emotiv Unity Plugin
                private void OnBandPowerDataReceived(object sender, ArrayList e)
                {
                    string dataText = "pow data: ";
                    foreach (var item in e)
                    {
                        dataText += item.ToString() + ";";
                    }

                    BandPowerDataMessage = dataText;
                    // print out data to console
                    //UnityEngine.Debug.Log(dataText);
                }

                private void OnPerfDataReceived(object sender, ArrayList e)
                {
                    string dataText = "met data: ";
                    foreach (var item in e)
                    {
                        dataText += item.ToString() + ";";
                    }

                    PerformanceMetricsDataMessage = dataText;
                    // print out data to console
                    //UnityEngine.Debug.Log(dataText);
                }

                private void OnMotionDataReceived(object sender, ArrayList e)
                {
                    string dataText = "mot data: ";
                    foreach (var item in e)
                    {
                        dataText += item.ToString() + ";";
                    }

                    MotionDataMessage = dataText;
                    // print out data to console
                    //UnityEngine.Debug.Log(dataText);
                }

                private void OnEEGDataReceived(object sender, ArrayList e)
                {
                    string dataText = "eeg data: ";
                    foreach (var item in e)
                    {
                        dataText += item.ToString() + ";";
                    }

                    RawEegDataMessage = dataText;

                    string path = _path + _rawEEGFileName;

                    if (!File.Exists(path))//"EmotivCortexBCI.txt"))
                    {
                        // Create a file to write to.
                        using (StreamWriter sw = File.CreateText(path))//"BitalinoResults.txt"))
                        {
                            sw.WriteLine(dataText);
                        }
                    }
                    else
                    {
                        // This text is always added, making the file longer over time
                        // if it is not deleted.
                        using (StreamWriter sw = File.AppendText(path))//"BitalinoResults.txt"))
                        {
                            sw.WriteLine(dataText);
                        }
                    }

                    // ToDo: to get all samples with sampling rate of 2000 hz needs to create a streamwrite here

                    // print out data to console
                    //UnityEngine.Debug.Log(dataText);
                }

                private void OnSysEventsReceived(object sender, SysEventArgs data)
                {
                    string dataText = "sys data: " + data.Detection + "; event: " + data.EventMessage + "; time " + data.Time.ToString();
                    _sysEventDataMessage = dataText;
                    // print out data to console
                    UnityEngine.Debug.Log(dataText);                    
                }

                private void OnDevDataReceived(object sender, ArrayList e)
                {
                    string dataText = "dev data: ";
                    foreach (var item in e)
                    {
                        dataText += item.ToString() + ";";
                    }

                    DevDataMessage = dataText;
                    // print out data to console
                    //UnityEngine.Debug.Log(dataText);
                }

                private void OnMentalCommandReceived(object sender, MentalCommandEventArgs data)
                {
                    string dataText = "com data: " + data.Act + ", power: " + data.Pow.ToString() + ", time " + data.Time.ToString();
                    // print out data to console
                    UnityEngine.Debug.Log(dataText);
                }

                private void OnFacialExpReceived(object sender, FacEventArgs data)
                {
                    string dataText = "Facial data: eye act " + data.EyeAct + ", upper act: " +
                                        data.UAct + ", upper act power " + data.UPow.ToString() + ", lower act: " +
                                        data.LAct + ", lower act power " + data.LPow.ToString() + ", time: " + data.Time.ToString();
                    // print out data to console
                    UnityEngine.Debug.Log(dataText);
                }

                public string EmotivAppslicationPath()
                {
                    string path = Application.dataPath;
                    string newPath = "";
                    if (Application.platform == RuntimePlatform.OSXPlayer)
                    {
                        newPath = Path.GetFullPath(Path.Combine(path, @"../../"));
                    }
                    else if (Application.platform == RuntimePlatform.WindowsPlayer)
                    {
                        newPath = Path.GetFullPath(Path.Combine(path, @"../"));
                    }
                    return newPath;
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
