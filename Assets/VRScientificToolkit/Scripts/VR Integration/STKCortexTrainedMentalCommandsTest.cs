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
    public class STKCortexTrainedMentalCommandsTest : MonoBehaviour
    {
        public string _mentalCommandProfileName;
        private bool _isProfileLoaded = false;
        public GameObject _mentalCommandTestObject;
        private Vector3 _mentalCommandTestObjectPosition;


        private enum DataStreamType { Motion, EEG, PerformanceMetrics, BandPowerData };

        [SerializeField]
        public EmotivParameter _isMentalCommandTracking = new EmotivParameter();
        private bool _previousIsMentalCommandTrackingValue;

        public void Awake()
        {
            _isMentalCommandTracking.name = "_isMentalCommandTracking";
            _isMentalCommandTracking.hideFromInspector = true;
            _isMentalCommandTracking.value = false;
            _previousIsMentalCommandTrackingValue = false;
        }

        ConnectToCortexStates _lastState;

        private DataStreamManager _dataStream = DataStreamManager.Instance;
        private Logger _logger = Logger.Instance;
        private HeadsetFinder _headsetFinder = HeadsetFinder.Instance;
        private TrainingHandler _trainingHandler = TrainingHandler.Instance;
        private Headset _headsetInformation;
        private BCITraining _bciTraining;

        //float _timerCortex_state = 0;
        const float TIME_UPDATE_CORTEX_STATE = 0.5f;

        //float _timerCounter_CQ = 0;
        float _timerCounter_queryHeadset = 0;
        //float _timerCounter_LicenseValidDay = 0;

        double _nRemainingDay = -1;

        const float TIME_UPDATE_CQ = 0.5f;
        const float TIME_QUERY_HEADSET = 2.0f;
        const float TIME_LICENSE_VALID_DAY = 1.0f;
        const float TIME_UPDATE_DATA = 1f;

        float _timerDataUpdate = 0;

        bool _isConnectDone = false;
        //bool _isConnected = false;
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
            _dataStream.MentalCommandReceived += OnMentalCommandReceived;
            
            _trainingHandler.ProfileLoaded += OnProfileLoaded;

            _mentalCommandTestObjectPosition = _mentalCommandTestObject.transform.position;

            //_mentalCommandTestCube = GameObject.Find("MentalCommandTestCube");
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

                    _timerDataUpdate += Time.deltaTime;
                    if (_timerDataUpdate < TIME_UPDATE_DATA)
                        return;

                    _timerDataUpdate -= TIME_UPDATE_DATA;

                    DevTracking();

                    bool testTageStarted = STKTestStage.GetStarted();

                    if (testTageStarted)
                    {

                        _mentalCommandTestObject.transform.position = _mentalCommandTestObjectPosition;

                        Debug.Log("Loading profile: " + _mentalCommandProfileName);
                        if (!_isProfileLoaded)
                        {
                            _bciTraining.QueryProfile();
                            List<string> profileList = _bciTraining.ProfileLists;

                            if (profileList != null && profileList.Contains(_mentalCommandProfileName))
                            {
                                //_bciTraining.UnLoadProfile(_mentalCommandProfileName);
                                //Thread.Sleep(1000);
                                _bciTraining.LoadProfile(_mentalCommandProfileName);
                                Thread.Sleep(1000);
                            }
                            else
                                Debug.Log("Could not load profile: " + _mentalCommandProfileName + " because it does not exists!");
                        }
                        else
                        {
                            _isMentalCommandTracking.hideFromInspector = false;

                            if (_isMentalCommandTracking.value != _previousIsMentalCommandTrackingValue)
                            {
                                _previousIsMentalCommandTrackingValue = _isMentalCommandTracking.value;

                                List<string> dataStreamList = new List<string>() { DataStreamName.MentalCommands };
                                if (_isMentalCommandTracking.value == true)
                                    _dataStream.SubscribeMoreData(dataStreamList);
                                else
                                    _dataStream.UnSubscribeData(dataStreamList);
                            }
                        }
                    }
                    else
                    {
                        if (_isMentalCommandTracking.value)
                        {
                            List<string> dataStreamList = new List<string>() { DataStreamName.MentalCommands };
                            _dataStream.UnSubscribeData(dataStreamList);
                            _previousIsMentalCommandTrackingValue = false;
                            _isMentalCommandTracking.value = false;
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
            List<string> dataStreamList = new List<string>() { DataStreamName.DevInfos, DataStreamName.SysEvents };
            _dataStream.StartDataStream(dataStreamList, headset.HeadsetID);
        }

        private void SessionActivatedOK(object sender, string headsetID)
        {
            Debug.Log("======================== SessionActivatedOK : " + headsetID);
            _isSessionActivated = true;
            
            _bciTraining = new BCITraining();
            _bciTraining.Init();
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

        private void OnMentalCommandReceived(object sender, MentalCommandEventArgs mentalCommandEventArgs)
        {
            // mentalcommand actions:
            // "neutral", "push", "pull", "lift", "drop", "left", "right", "rotateLeft", "rotateRight", "rotateClockwise", "rotateCounterClockwise", "rotateForwards", "rotateReverse", "disappear"

            if (mentalCommandEventArgs.Act.Equals("push"))
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.x -= 1.0f;
            }
            else if (mentalCommandEventArgs.Act.Equals("pull")) 
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.x += 1.0f;
            }
            else if (mentalCommandEventArgs.Act.Equals("lift"))
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.y += 1.0f;
            }
            else if (mentalCommandEventArgs.Act.Equals("drop"))
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.y -= 1.0f;
            }              
            else if (mentalCommandEventArgs.Act.Equals("left"))
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.z += 1.0f;
            }
            else if (mentalCommandEventArgs.Act.Equals("right"))
            {
                Debug.Log("STKOnMentalCommandReceived: action = " + mentalCommandEventArgs.Act + " power = " + mentalCommandEventArgs.Pow);
                _mentalCommandTestObjectPosition.z -= 1.0f;
            }
            else
            {
                Debug.Log("STKOnMentalCommandReceived: Unhandeld mentalcommand!");
            }
        }

        private void OnProfileLoaded(object sender, string profileName)
        {
            Debug.Log("STKBCITraining: OnProfileLoaded profile " + profileName);
            _isProfileLoaded = true;
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
