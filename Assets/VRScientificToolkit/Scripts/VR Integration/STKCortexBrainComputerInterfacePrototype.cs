using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Text;
using System.IO;
using System;
using Unity;
using UnityEngine.UI;
using UnityEditor;
using UnityEngine;
using EmotivUnityPlugin;
using Newtonsoft.Json.Linq;

namespace STK
{
    ///<summary>Property drawer for defining Emotiv Cortex Event Parameters. Can also block the changing of Events that were Auto-generated</summary>
    [CustomPropertyDrawer(typeof(EmotivParameter))]
    public class STKEmotivEditor : PropertyDrawer
    {

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {

            if (property.FindPropertyRelative("hideFromInspector").boolValue == false)
            {
                EditorGUI.PropertyField(new Rect(position.x, position.y, position.width, 17), property.FindPropertyRelative("name"));
                EditorGUI.PropertyField(new Rect(position.x, position.y + 20f, position.width, 17), property.FindPropertyRelative("value"));
            }
            else
            {
                EditorGUI.LabelField(new Rect(position.x, position.y, position.width, 17), property.FindPropertyRelative("name").stringValue);
                EditorGUI.LabelField(new Rect(position.x, position.y, position.width, 17), property.FindPropertyRelative("value").stringValue);
            }

        }

        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            return 50.0f;
        }

    }

    /// <summary>
    /// Defines the parameter of an Emotiv Cortex Event handlingg. 
    /// </summary>
    [System.Serializable]
    public class EmotivParameter
    {
        public string name;
        public bool hideFromInspector;
        public bool value;
    }

    public class STKCortexBrainComputerInterfacePrototype : MonoBehaviour
    {
        private enum DataStreamType { Motion, EEG, PerformanceMetrics };

        [SerializeField]
        public EmotivParameter _isMotionTracking = new EmotivParameter();
        private bool _previousIsMotionTrackingValue;

        [SerializeField]
        public EmotivParameter _isEegTrackingActive = new EmotivParameter();
        private bool _previousIsEegTrackingActiveValue;

        [SerializeField]
        public EmotivParameter _isPerformanceMetricsTrackingActive = new EmotivParameter();
        private bool _previousIsPerformanceMetricsTrackingActiveValue;

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
        }

        ConnectToCortexStates _lastState;

        DataStreamManager _dataStream = DataStreamManager.Instance;
        Logger _logger = Logger.Instance;
        private HeadsetFinder _headsetFinder = HeadsetFinder.Instance;
        private Headset _headsetInformation;

        float _timerCortex_state = 0;
        const float TIME_UPDATE_CORTEX_STATE = 0.5f;

        float _timerCounter_CQ = 0;
        float _timerCounter_queryHeadset = 0;
        float _timerCounter_LicenseValidDay = 0;

        double _nRemainingDay = -1;

        const float TIME_UPDATE_CQ = 0.5f;
        const float TIME_QUERY_HEADSET = 2.0f;
        const float TIME_LICENSE_VALID_DAY = 1.0f;
        const float TIME_UPDATE_DATA = 1f;

        float _timerDataUpdate = 0;

        bool _isConnectDone = false;
        bool _isConnected = false; 
        bool _isQueryHeadset = false;
        bool _isQueryHeadsetDeactivated = false;
        bool _isConnectedToDevice = false;
        bool _isSessionActivated = false;
    
        // Start is called before the first frame update
        void Start()
        {
            _dataStream.SetAppConfig(STKCortexAppConfig.ClientId, STKCortexAppConfig.ClientSecret,
                                     STKCortexAppConfig.AppVersion, STKCortexAppConfig.AppName,
                                     STKCortexAppConfig.TmpAppDataDir, STKCortexAppConfig.AppUrl,
                                     EmotivAppslicationPath());

            // Init logger
            _logger.Init();

            Debug.Log("Configure PRODUCT SERVER - version: " + STKCortexAppConfig.AppVersion);

            // start App
            _dataStream.StartAuthorize(STKCortexAppConfig.AppLicenseId);

            _dataStream.SessionActivatedOK += SessionActivatedOK;
            _dataStream.LicenseValidTo += OnLicenseValidTo;
            
        }

        // Update is called once per frame
        void Update()
        {

            if (_isConnectDone)
            {
                if (_isQueryHeadset)
                {
                    Debug.Log("=============== Count GetDetectedHeadsets : " + _dataStream.GetDetectedHeadsets().Count);
                    if (_dataStream.GetDetectedHeadsets().Count > 0 && !_isConnectedToDevice)
                    {
                        Debug.Log("=============== Count GetDetectedHeadsets : " + _dataStream.GetDetectedHeadsets().Count);
                        startConnectToDevice();
                        _isConnectedToDevice = true;
                        
                    }

                    _timerCounter_queryHeadset += Time.deltaTime;
                    if (_timerCounter_queryHeadset > TIME_QUERY_HEADSET && !_isSessionActivated)
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

                if (_isSessionActivated)
                {
                    
                    if (!_isQueryHeadsetDeactivated)
                    {
                        _headsetFinder.StopQueryHeadset();
                        _isQueryHeadsetDeactivated = true;
                    }

                    if (_isMotionTracking.value != _previousIsMotionTrackingValue)
                    {
                        _previousIsMotionTrackingValue = _isMotionTracking.value;

                        List<string> dataStreamList = new List<string>() { DataStreamName.Motion };
                        if (_isMotionTracking.value == true)
                            _dataStream.SubscribeMoreData(dataStreamList);
                        else
                            _dataStream.UnSubscribeData(dataStreamList);
                    }

                    if (_isPerformanceMetricsTrackingActive.value != _previousIsPerformanceMetricsTrackingActiveValue)
                    {
                        _previousIsPerformanceMetricsTrackingActiveValue = _isPerformanceMetricsTrackingActive.value;

                        List<string> dataStreamList = new List<string>() { DataStreamName.PerformanceMetrics };
                        if (_isPerformanceMetricsTrackingActive.value == true)
                            _dataStream.SubscribeMoreData(dataStreamList);
                        else
                            _dataStream.UnSubscribeData(dataStreamList);
                    }

                    if (_isEegTrackingActive.value != _previousIsEegTrackingActiveValue)
                    {
                        _previousIsEegTrackingActiveValue = _isEegTrackingActive.value;

                        List<string> dataStreamList = new List<string>() { DataStreamName.EEG };
                        if (_isEegTrackingActive.value == true)
                            _dataStream.SubscribeMoreData(dataStreamList);
                        else
                            _dataStream.UnSubscribeData(dataStreamList);
                    }

                    _timerDataUpdate += Time.deltaTime;
                    if (_timerDataUpdate < TIME_UPDATE_DATA)
                        return;

                    _timerDataUpdate -= TIME_UPDATE_DATA;

                    DevTracking();


                    bool testTageStarted = STKTestStage.GetStarted();

                    if (testTageStarted)
                    {
                        _isMotionTracking.hideFromInspector = false;
                        _isEegTrackingActive.hideFromInspector = false;
                        _isPerformanceMetricsTrackingActive.hideFromInspector = false;

                        if (_isMotionTracking.value)
                            MotionTracking();

                        if (_isEegTrackingActive.value)
                            EegTracking();

                        if (_isPerformanceMetricsTrackingActive.value)
                            PerformanceMatricTracking();
                    }
                    else
                    {
                        if (_isMotionTracking.value)
                        {
                            List<string> dataStreamList = new List<string>() { DataStreamName.Motion };
                            _dataStream.UnSubscribeData(dataStreamList);
                            _previousIsMotionTrackingValue = false;
                            _isMotionTracking.value = false;
                        }

                        if (_isEegTrackingActive.value)
                        {
                            List<string> dataStreamList = new List<string>() { DataStreamName.EEG };
                            _dataStream.UnSubscribeData(dataStreamList);
                            _isEegTrackingActive.value = false;
                            _previousIsEegTrackingActiveValue = false;
                        }

                        if (_isPerformanceMetricsTrackingActive.value)
                        {
                            List<string> dataStreamList = new List<string>() { DataStreamName.PerformanceMetrics };
                            _dataStream.UnSubscribeData(dataStreamList);
                            _isPerformanceMetricsTrackingActive.value = false;
                            _previousIsPerformanceMetricsTrackingActiveValue = false;
                        }
                    }
                }

                return;
            }

            var curState = _dataStream.GetConnectToCortexState();

            switch (curState)
            {
                case ConnectToCortexStates.Service_connecting:
                    {
                        Debug.Log("=============== Connecting To service");
                        break;
                    }
                case ConnectToCortexStates.EmotivApp_NotFound:
                    {

                        _isConnectDone = true;
                        Debug.Log("=============== Connect_failed + EmotivApp_NotFound");
                        break;
                    }
                case ConnectToCortexStates.Login_waiting:
                    {
                        Debug.Log("=============== Login_waiting");
                        break;
                    }
                case ConnectToCortexStates.Login_notYet:
                    {
                        _isConnectDone = true;
                        Debug.Log("=============== Login_notYet");
                        break;
                    }
                case ConnectToCortexStates.Authorizing:
                    {
                        Debug.Log("=============== Authenticating...");
                        break;
                    }
                case ConnectToCortexStates.Authorize_failed:
                    {
                        Debug.Log("=============== Authorize_failed");
                        break;
                    }
                case ConnectToCortexStates.Authorized:
                    {
                        _isConnectDone = true;
                        Debug.Log("=============== Authorized");
                        break;
                    }
                case ConnectToCortexStates.LicenseExpried:
                    {
                        _isConnectDone = true;
                        Debug.Log("=============== Trial expired");
                        break;
                    }
                case ConnectToCortexStates.License_HardLimited:
                    {
                        _isConnectDone = true;
                        Debug.Log("=============== License_HardLimited");
                        break;
                    }
            }

        }

        void OnDestroy()
        {
            _dataStream.Stop();
        }

        private void startConnectToDevice()
        {
            _headsetInformation = _dataStream.GetDetectedHeadsets()[0];
            Headset headset = _dataStream.GetDetectedHeadsets()[0];
            List<string> dataStreamList = new List<string>() { DataStreamName.DevInfos };
            _dataStream.StartDataStream(dataStreamList, headset.HeadsetID);
        }

        private void SessionActivatedOK(object sender, string headsetID)
        {
            Debug.Log("======================== SessionActivatedOK : " + headsetID);
            _isSessionActivated = true;
        }

        private void OnLicenseValidTo(object sender, DateTime validToDate)
        {
            DateTime curUTC_Now = DateTime.UtcNow;
            System.TimeSpan diffDate = validToDate - curUTC_Now;
            double nDay = (double)((int)(diffDate.TotalDays * 10)) / 10;
            if (nDay < 0)
                _nRemainingDay = 0;
            else
                _nRemainingDay = nDay;

            Debug.Log("ValidToDate = " + validToDate + ", UTC = " + curUTC_Now + ", nDay = " + nDay);
        }
        
        private void MotionTracking()
        {
            if (_dataStream.GetNumberMotionSamples() > 0)
            {
                string motHeaderStr = string.Empty;
                string motDataStr = string.Empty;
                foreach (var ele in _dataStream.GetMotionChannels())
                {
                    string chanStr = ChannelStringList.ChannelToString(ele);
                    double[] data = _dataStream.GetMotionData(ele);
                    motHeaderStr += chanStr + ", ";
                    if (data != null && data.Length > 0)
                        motDataStr += data[0].ToString() + ", ";
                }
                if (!motDataStr.Equals(string.Empty))
                {
                    DeployReceivedStreamDataHeader(DataStreamType.Motion, motHeaderStr);
                    DeployReceivedStreamData(DataStreamType.Motion, motDataStr);
                }

                // string title, float value, Color color = default
                //DrawGraph.Add("title", 33.3f);

                // only as an playback example
                //{
                //    GameObject motionHeaderTextField = GameObject.Find("EmotivDataStreamHeaderEvent");
                //    if (motionHeaderTextField != null)
                //        motionHeaderTextField.GetComponent<Text>().text = motHeaderStr;

                //    GameObject motionDataTextField = GameObject.Find("EmotivDataStreamEvent");
                //    if (motionDataTextField != null)
                //        motionDataTextField.GetComponent<Text>().text = motDataStr;

                //}

                Debug.Log(" Motion Header : " + motHeaderStr);
                Debug.Log(" Motion Data : " + motDataStr);
            }
        }

        private void EegTracking()
        {
            // update EEG data
            if (_dataStream.GetNumberEEGSamples() > 0)
            {
                string eegHeaderStr = string.Empty;
                string eegDataStr = string.Empty;
                foreach (var ele in _dataStream.GetEEGChannels())
                {
                    string chanStr = ChannelStringList.ChannelToString(ele);
                    double[] data = _dataStream.GetEEGData(ele);
                    eegHeaderStr += chanStr + ", ";
                    if (data != null && data.Length > 0)
                        eegDataStr += data[0].ToString() + ", ";
                }

                //if (!eegDataStr.Equals(string.Empty))
                //{
                //    DeployReceivedStreamDataHeader(eegHeaderStr);
                //    DeployReceivedStreamData(eegDataStr);
                //}
                Debug.Log("EEG Header: " + eegHeaderStr);
                Debug.Log("EEG Data: " + eegDataStr);
            }
        }

        private void PerformanceMatricTracking()
        {
            // update pm data
            if (_dataStream.GetNumberPMSamples() > 0)
            {
                string pmHeaderStr = string.Empty;
                string pmDataStr = string.Empty;
                bool hasPMUpdate = true;
                foreach (var ele in _dataStream.GetPMLists())
                {
                    string chanStr = ele;
                    double data = _dataStream.GetPMData(ele);
                    if (chanStr == "TIMESTAMP" && (data == -1))
                    {
                        // has no new update of performance metric data
                        hasPMUpdate = false;
                        Debug.Log("PerformanceMatricTracking: has no new update of performance metric data");
                        break;
                    }
                    pmHeaderStr += chanStr + ", ";
                    pmDataStr += data.ToString() + ", ";
                }
                if (hasPMUpdate)
                {
                    DeployReceivedStreamDataHeader(DataStreamType.PerformanceMetrics, pmHeaderStr);
                    DeployReceivedStreamData(DataStreamType.PerformanceMetrics, pmDataStr);

                    Debug.Log("Performance metrics Header: " + pmHeaderStr);
                    Debug.Log("Performance metrics Data: " + pmDataStr);
                }

            }
        }

        private void DevTracking()
        {
            if (_dataStream.GetNumberCQSamples() > 0)
            {
                float _currentBatteryLevel = (float)DataStreamManager.Instance.Battery();
                int _maxBatteryLevel = (int)DataStreamManager.Instance.BatteryMax();

                Debug.LogFormat("charge level (%) : {0}, max battery level: {1}", _currentBatteryLevel, _maxBatteryLevel);
                Debug.Log("SignalStrength : " + _dataStream.SignalStrength());
            }
        }

        private void DeployReceivedStreamDataHeader(DataStreamType dataStreamType, string data)
        {
            switch (dataStreamType)
            {
                case DataStreamType.Motion:
                    {
                        GetComponents<STKEventSender>()[0].SetEventValue("text_Text", data);
                        GetComponents<STKEventSender>()[0].Deploy();
                        break;
                    }
                case DataStreamType.PerformanceMetrics:
                    {
                        GetComponents<STKEventSender>()[2].SetEventValue("text_Text", data);
                        GetComponents<STKEventSender>()[2].Deploy();
                        break;
                    }
            }
        }

        private void DeployReceivedStreamData(DataStreamType dataStreamType, string data)
        {
            switch (dataStreamType)
            {
                case DataStreamType.Motion:
                    {
                        GetComponents<STKEventSender>()[1].SetEventValue("text_Text", data);
                        GetComponents<STKEventSender>()[1].Deploy();
                        break;
                    }
                case DataStreamType.PerformanceMetrics:
                    {
                        GetComponents<STKEventSender>()[3].SetEventValue("text_Text", data);
                        GetComponents<STKEventSender>()[3].Deploy();
                        break;
                    }
            }
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

    }
}
