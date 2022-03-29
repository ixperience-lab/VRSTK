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
    public class STKCortexBrainComputerInterfaceTraining : MonoBehaviour
    {
        public string _mentalCommandProfileName;
        private string _currentAction;
        private string _currentDetektion = "mentalCommand";
        private bool _isProfileLoaded = false;
        private bool _isSucceeded = false;

        ConnectToCortexStates _lastState;

        DataStreamManager _dataStream = DataStreamManager.Instance;
        Logger _logger = Logger.Instance;
        private HeadsetFinder _headsetFinder = HeadsetFinder.Instance;
        private TrainingHandler _trainingHandler = TrainingHandler.Instance;
        private BCITraining _bciTraining;
        private Headset _headsetInformation;

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

            _trainingHandler.CreateProfileOK += OnCreateProfileOK;
            _trainingHandler.TrainingOK += OnTrainingOK;
            _trainingHandler.GetDetectionInfoOK += OnGetDetectionInfoOK;
            _trainingHandler.ProfileLoaded += OnProfileLoaded;
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
                       
                        //ConsoleKeyInfo keyInfo = Console.ReadKey(true);

                        //Debug.Log(keyInfo.KeyChar.ToString() + " has pressed");
                        
                        _bciTraining.QueryProfile();
                        List<string> profileList = _bciTraining.ProfileLists;

                        //if (keyInfo.Key == ConsoleKey.C)
                        Debug.Log("Key C is pressed: " + Input.GetKey(KeyCode.C));
                        Debug.Log("Key L is pressed: " + Input.GetKey(KeyCode.L));
                        Debug.Log("Key A is pressed: " + Input.GetKey(KeyCode.A));
                        Debug.Log("Key 0 is pressed: " + Input.GetKey(KeyCode.Alpha0));
                        Debug.Log("Key 1 is pressed: " + Input.GetKey(KeyCode.Alpha1));
                        Debug.Log("Key 2 is pressed: " + Input.GetKey(KeyCode.Alpha2));
                        Debug.Log("Key 3 is pressed: " + Input.GetKey(KeyCode.Alpha3));
                        Debug.Log("Key 4 is pressed: " + Input.GetKey(KeyCode.Alpha4));
                        Debug.Log("Key 5 is pressed: " + Input.GetKey(KeyCode.Alpha5));
                        Debug.Log("Key 6 is pressed: " + Input.GetKey(KeyCode.Alpha6));

                        if (Input.GetKey(KeyCode.C))
                        {
                            if (string.IsNullOrEmpty(_mentalCommandProfileName))
                                _mentalCommandProfileName = Utils.GenerateUuidProfileName("McDemo");

                            Debug.Log("Creating profile: " + _mentalCommandProfileName);

                            if (profileList != null && profileList.Contains(_mentalCommandProfileName))
                                Debug.Log("Could not create profile: " + _mentalCommandProfileName + " because it already exists!");
                            else
                            {
                                _bciTraining.CreateProfile(_mentalCommandProfileName);
                                Thread.Sleep(1000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.L))//if (keyInfo.Key == ConsoleKey.L)
                        {
                            //Load profile
                            Debug.Log("Loading profile: " + _mentalCommandProfileName);
                            if (profileList != null && profileList.Contains(_mentalCommandProfileName))
                            {
                                _bciTraining.LoadProfile(_mentalCommandProfileName);
                                Thread.Sleep(1000);
                            }
                            else
                                Debug.Log("Could not load profile: " + _mentalCommandProfileName + " because it does not exists!");
                        }
                        else if (Input.GetKey(KeyCode.U))//if (keyInfo.Key == ConsoleKey.U)
                        {
                            //Load profile
                            Debug.Log("UnLoad profile: " + _mentalCommandProfileName);
                            _bciTraining.UnLoadProfile(_mentalCommandProfileName);
                            Thread.Sleep(1000);
                        }
                        else if (Input.GetKey(KeyCode.Alpha0))//if (keyInfo.Key == ConsoleKey.D0)
                        {
                            if (_isProfileLoaded)
                            {
                                _currentAction = "neutral";
                                //Start neutral training
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha1))//if (keyInfo.Key == ConsoleKey.D1)
                        {
                            if (_isProfileLoaded)
                            {
                                //Start push training
                                _currentAction = "push";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha2))//if (keyInfo.Key == ConsoleKey.D2)
                        {
                            if (_isProfileLoaded)
                            {
                                //Start pull training
                                _currentAction = "pull";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha3))
                        {
                            if (_isProfileLoaded)
                            {
                                //Start pull training
                                _currentAction = "lift";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha4))
                        {
                            if (_isProfileLoaded)
                            {
                                //Start pull training
                                _currentAction = "drop";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha5))
                        {
                            if (_isProfileLoaded)
                            {
                                //Start pull training
                                _currentAction = "left";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.Alpha6))
                        {
                            if (_isProfileLoaded)
                            {
                                //Start pull training
                                _currentAction = "right";
                                _bciTraining.StartTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(2000);
                            }
                        }
                        else if (Input.GetKey(KeyCode.A))//if (keyInfo.Key == ConsoleKey.A)
                        {
                            //Accept training
                            if (_isSucceeded)
                            {
                                _bciTraining.AcceptTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(1000);
                                _isSucceeded = false; // reset
                            }
                        }
                        else if (Input.GetKey(KeyCode.R))//if (keyInfo.Key == ConsoleKey.R)
                        {
                            //Reject training
                            if (_isSucceeded)
                            {
                                _bciTraining.RejectTraining(_currentAction, _currentDetektion);
                                Thread.Sleep(1000);
                                _isSucceeded = false; // reset
                            }
                        }
                        else if (Input.GetKey(KeyCode.H))//if (keyInfo.Key == ConsoleKey.H)
                        {
                            Debug.Log("Press C to create a Profile.");
                            Debug.Log("Press L to load a Profile.");
                            Debug.Log("Press U to unload a Profile.");
                            Debug.Log("Press Alpha0 to start Neutral training.");
                            Debug.Log("Press Alpha1 to start Push training.");
                            Debug.Log("Press Alpha2 to start Pull training.");
                            Debug.Log("Press Alpha3 to start Lift training.");
                            Debug.Log("Press Alpha4 to start Drop training.");
                            Debug.Log("Press Alpha5 to start Left training.");
                            Debug.Log("Press Alpha6 to start Right training.");
                            Debug.Log("Press A to accept training.");
                            Debug.Log("Press R to reject training.");
                            Debug.Log("Press H to show all commands");
                        }
                        else
                        {
                            Debug.Log("Unsupported key");
                        }

                        
                        // Or start a training. You have to subscribe "sys" and "com" for mental command or "fac" for facial expression
                        // mentalcommand actions:
                        // "neutral", "push", "pull", "lift", "drop", "left", "right", "rotateLeft", "rotateRight", "rotateClockwise", "rotateCounterClockwise", "rotateForwards", "rotateReverse", "disappear"

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
            List<string> dataStreamList = new List<string>() { DataStreamName.DevInfos, DataStreamName.SysEvents };//, DataStreamName.MentalCommands };
            _dataStream.StartDataStream(dataStreamList, headset.HeadsetID);
        }

        private void SessionActivatedOK(object sender, string headsetID)
        {
            Debug.Log("======================== SessionActivatedOK : " + headsetID);
            _isSessionActivated = true;

            _bciTraining = new BCITraining();
            _bciTraining.Init();

            Debug.Log("Press C to create a Profile.");
            Debug.Log("Press L to load a Profile.");
            Debug.Log("Press U to unload a Profile.");
            Debug.Log("Press Alpha0 to start Neutral training.");
            Debug.Log("Press Alpha1 to start Push training.");
            Debug.Log("Press Alpha2 to start Pull training.");
            Debug.Log("Press Alpha3 to start Lift training.");
            Debug.Log("Press Alpha4 to start Drop training.");
            Debug.Log("Press Alpha5 to start Left training.");
            Debug.Log("Press Alpha6 to start Right training.");
            Debug.Log("Press A to accept training.");
            Debug.Log("Press R to reject training.");
            Debug.Log("Press H to show all commands");
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

        private void OnCreateProfileOK(object sender, string profileName)
        {
            Debug.Log("STKBCITraining: OnCreateProfileOK profilename " + profileName);
            _isProfileLoaded = true;
        }

        private void OnTrainingOK(object sender, JObject result)
        {
            Debug.Log("STKBCITraining:  TrainingOK = " + result);
            _isSucceeded = true;
        }

        private void OnProfileLoaded(object sender, string profileName)
        {
            Debug.Log("STKBCITraining: OnProfileLoaded profile " + profileName);
            _isProfileLoaded = true;
        }

        private void OnGetDetectionInfoOK(object sender, DetectionInfo detectionInfo)
        {
            if (detectionInfo.DetectionName == _currentDetektion)
            {
                Debug.Log("OnGetDetectionInfoOK currentDectektion: " + _currentDetektion);

                // QueryProfile
                //QueryProfile();
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