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
using TMPro;
using VRSTK.Scripts.Telemetry;
using VRSTK.Scripts.TestControl;

namespace VRSTK
{
    namespace Scripts
    {
        namespace VRIntegration
        {
            ///<summary>Property drawer for defining Emotiv Cortex Event Parameters. Can also block the changing of Events that were Auto-generated</summary>
            [CustomPropertyDrawer(typeof(EmotivParameter))]
            public class EmotivEditor : PropertyDrawer
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
                    }

                }

                public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
                {
                    return 50.0f;
                }

            }


            ///<summary>Property drawer for defining Emotiv Cortex Event Parameters. Can also block the changing of Events that were Auto-generated</summary>
            [CustomPropertyDrawer(typeof(EmotivGraphParameter))]
            public class EmotivGraphEditor : PropertyDrawer
            {
                private Material material = new Material(Shader.Find("Hidden/Internal-Colored"));

                public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
                {

                    // Begin to draw a horizontal layout, using the helpBox EditorStyle
                    GUILayout.BeginHorizontal(EditorStyles.helpBox);

                    // Reserve GUI space with a width from 10 to 10000, and a fixed height of 200, and 
                    // cache it as a rectangle.
                    Rect layoutRectangle = GUILayoutUtility.GetRect(10, 10000, 200, 200);

                    EditorGUI.PropertyField(new Rect(layoutRectangle.x, layoutRectangle.y, layoutRectangle.width, layoutRectangle.height), property.FindPropertyRelative("name"));

                    if (UnityEngine.Event.current.type == EventType.Repaint)
                    {
                        // If we are currently in the Repaint event, begin to draw a clip of the size of 
                        // previously reserved rectangle, and push the current matrix for drawing.
                        GUI.BeginClip(layoutRectangle);
                        GL.PushMatrix();

                        // Clear the current render buffer, setting a new background colour, and set our
                        // material for rendering.
                        GL.Clear(true, false, Color.black);
                        material.SetPass(0);

                        // Start drawing in OpenGL Quads, to draw the background canvas. Set the
                        // colour black as the current OpenGL drawing colour, and draw a quad covering
                        // the dimensions of the layoutRectangle.
                        GL.Begin(GL.QUADS);
                        GL.Color(Color.black);
                        GL.Vertex3(0, 0, 0);
                        GL.Vertex3(layoutRectangle.width, 0, 0);
                        GL.Vertex3(layoutRectangle.width, layoutRectangle.height, 0);
                        GL.Vertex3(0, layoutRectangle.height, 0);
                        GL.End();

                        // Start drawing in OpenGL Lines, to draw the lines of the grid.
                        GL.Begin(GL.LINES);

                        // Store measurement values to determine the offset, for scrolling animation,
                        // and the line count, for drawing the grid.
                        int offset = (Time.frameCount * 2) % 50;
                        int count = (int)(layoutRectangle.width / 10) + 20;

                        for (int i = 0; i < count; i++)
                        {
                            // For every line being drawn in the grid, create a colour placeholder; if the
                            // current index is divisible by 5, we are at a major segment line; set this
                            // colour to a dark grey. If the current index is not divisible by 5, we are
                            // at a minor segment line; set this colour to a lighter grey. Set the derived
                            // colour as the current OpenGL drawing colour.
                            Color lineColour = (i % 5) == 0 ? new Color(0.5f, 0.5f, 0.5f) : new Color(0.2f, 0.2f, 0.2f);
                            GL.Color(lineColour);

                            // Derive a new x co-ordinate from the initial index, converting it straight
                            // into line positions, and move it back to adjust for the animation offset.
                            float x = i * 10 - offset;

                            if (x >= 0 && x < layoutRectangle.width)
                            {
                                // If the current derived x position is within the bounds of the
                                // rectangle, draw another vertical line.
                                GL.Vertex3(x, 0, 0);
                                GL.Vertex3(x, layoutRectangle.height, 0);
                            }

                            if (i < layoutRectangle.height / 10)
                            {
                                // Convert the current index value into a y position, and if it is within
                                // the bounds of the rectangle, draw another horizontal line.
                                GL.Vertex3(0, i * 10, 0);
                                GL.Vertex3(layoutRectangle.width, i * 10, 0);
                            }
                        }

                        // End lines drawing.
                        GL.End();


                        GL.Begin(GL.LINES);
                        GL.Color(Color.red);
                        Vector3 _vec = property.FindPropertyRelative("value").vector3Value;
                        GL.Vertex3(_vec.x, _vec.y, 0);
                        GL.End();

                        // Pop the current matrix for rendering, and end the drawing clip.
                        GL.PopMatrix();
                        GUI.EndClip();
                    }

                    // End our horizontal 
                    GUILayout.EndHorizontal();

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

            /// <summary>
            /// Defines the parameter of an Emotiv Cortex Event handlingg. 
            /// </summary>
            [System.Serializable]
            public class EmotivGraphParameter
            {
                public string name;
                public bool hideFromInspector;
                public Vector3 value;
            }

            public class CortexBrainComputerInterfacePrototype : MonoBehaviour
            {
                double _engLastData = 0f;
                double _excLastData = 0f;
                double _lexLastData = 0f;
                double _strLastData = 0f;
                double _relLastData = 0f;
                double _intLastData = 0f;
                double _focLastData = 0f;


                private enum DataStreamType { Motion, EEG, PerformanceMetrics, BandPowerData };

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

                ConnectToCortexStates _lastState;

                DataStreamManager _dataStream = DataStreamManager.Instance;
                Logger _logger = Logger.Instance;
                private HeadsetFinder _headsetFinder = HeadsetFinder.Instance;
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
                    _dataStream.SetAppConfig(CortexAppConfig.ClientId, CortexAppConfig.ClientSecret,
                                             CortexAppConfig.AppVersion, CortexAppConfig.AppName,
                                             CortexAppConfig.TmpAppDataDir, CortexAppConfig.AppUrl,
                                             EmotivAppslicationPath());

                    // Init logger
                    _logger.Init();

                    Debug.Log("Configure PRODUCT SERVER - version: " + CortexAppConfig.AppVersion);

                    // start App
                    _dataStream.StartAuthorize(CortexAppConfig.AppLicenseId);

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

                            if (_isBandPowerTrackingActive.value != _previousIsBandPowerTrackingActiveValue)
                            {
                                _previousIsBandPowerTrackingActiveValue = _isBandPowerTrackingActive.value;

                                List<string> dataStreamList = new List<string>() { DataStreamName.BandPower };
                                if (_isBandPowerTrackingActive.value == true)
                                    _dataStream.SubscribeMoreData(dataStreamList);
                                else
                                    _dataStream.UnSubscribeData(dataStreamList);
                            }

                            _timerDataUpdate += Time.deltaTime;
                            if (_timerDataUpdate < TIME_UPDATE_DATA)
                                return;

                            _timerDataUpdate -= TIME_UPDATE_DATA;

                            DevTracking();


                            bool testTageStarted = TestStage.GetStarted();

                            if (testTageStarted)
                            {
                                _isMotionTracking.hideFromInspector = false;
                                _isEegTrackingActive.hideFromInspector = false;
                                _isPerformanceMetricsTrackingActive.hideFromInspector = false;
                                _isBandPowerTrackingActive.hideFromInspector = false;

                                if (_isMotionTracking.value)
                                    MotionTracking();

                                if (_isEegTrackingActive.value)
                                    EegTracking();

                                if (_isPerformanceMetricsTrackingActive.value)
                                    PerformanceMatricTracking();

                                if (_isBandPowerTrackingActive.value)
                                    BandPowerDataTracking();
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

                                if (_isBandPowerTrackingActive.value)
                                {
                                    List<string> dataStreamList = new List<string>() { DataStreamName.BandPower };
                                    _dataStream.UnSubscribeData(dataStreamList);
                                    _isBandPowerTrackingActive.value = false;
                                    _previousIsBandPowerTrackingActiveValue = false;
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

                        double engData = _engLastData;
                        double excData = _excLastData;
                        double lexData = _lexLastData;
                        double strData = _strLastData;
                        double relData = _relLastData;
                        double intData = _intLastData;
                        double focData = _focLastData;

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

                            if (chanStr.Equals("eng"))
                                engData = data;

                            if (chanStr.Equals("exc"))
                                excData = data;

                            if (chanStr.Equals("lex"))
                                lexData = data;

                            if (chanStr.Equals("str"))
                                strData = data;

                            if (chanStr.Equals("rel"))
                                relData = data;

                            if (chanStr.Equals("int"))
                                intData = data;

                            if (chanStr.Equals("foc"))
                                focData = data;

                            //eng, exc, lex, str, rel, int, foc, 

                            pmDataStr += data.ToString() + ", ";
                        }
                        if (hasPMUpdate)
                        {
                            GameObject dataRepresentationContainer = GameObject.Find("DataRepresentationContainer");
                            if (dataRepresentationContainer != null)
                                UpdateDataRepresentationContainer(engData, excData, lexData, strData, relData, intData, focData);

                            DeployReceivedStreamDataHeader(DataStreamType.PerformanceMetrics, pmHeaderStr);
                            DeployReceivedStreamData(DataStreamType.PerformanceMetrics, pmDataStr);

                            Debug.Log("Performance metrics Header: " + pmHeaderStr);
                            Debug.Log("Performance metrics Data: " + pmDataStr);
                        }

                    }
                }

                private void BandPowerDataTracking()
                {
                    string bandPowerDataHeaderStr = string.Empty;
                    string bandPowerDataStr = string.Empty;
                    if (_dataStream.GetNumberPowerBandSamples() > 0)
                    {
                        foreach (var ele in _dataStream.GetBandPowerLists())
                        {
                            string chanStr = ele;
                            if (chanStr.Equals("TIMESTAMP"))
                            {
                                bandPowerDataHeaderStr += chanStr + ", ";
                                bandPowerDataStr += _dataStream.GetThetaData(ChannelStringList.StringToChannel(chanStr)).ToString() + ", ";
                            }
                            else
                            {
                                string chanFromStringListEmelent = chanStr.Split('/')[0];
                                bandPowerDataHeaderStr += chanStr + ", ";
                                bandPowerDataStr += _dataStream.GetThetaData(ChannelStringList.StringToChannel(chanFromStringListEmelent)).ToString() + ", " + _dataStream.GetAlphaData(ChannelStringList.StringToChannel(chanFromStringListEmelent)).ToString() + ", " +
                                                    _dataStream.GetLowBetaData(ChannelStringList.StringToChannel(chanFromStringListEmelent)).ToString() + ", " + _dataStream.GetHighBetaData(ChannelStringList.StringToChannel(chanFromStringListEmelent)).ToString() + ", " +
                                                    _dataStream.GetGammaData(ChannelStringList.StringToChannel(chanFromStringListEmelent)).ToString() + ", ";
                            }
                        }

                        //if (!bandPowerDataStr.Equals(string.Empty))
                        //{
                        //    DeployReceivedStreamDataHeader(DataStreamType.BandPowerData, bandPowerDataHeaderStr);
                        //    DeployReceivedStreamData(DataStreamType.BandPowerData, bandPowerDataStr);
                        //}
                    }
                    Debug.Log("BandPowerData Header: " + bandPowerDataHeaderStr);
                    Debug.Log("BandPowerData Data: " + bandPowerDataStr);
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
                                GetComponents<EventSender>()[0].SetEventValue("text_Text", data);
                                GetComponents<EventSender>()[0].Deploy();
                                break;
                            }
                        case DataStreamType.PerformanceMetrics:
                            {
                                GetComponents<EventSender>()[2].SetEventValue("text_Text", data);
                                GetComponents<EventSender>()[2].Deploy();
                                break;
                            }
                    }
                }

                private void DeployReceivedStreamData(DataStreamType dataStreamType, string data)
                {

                    String[] splittedData = data.Split(',');
                    double timeStamp = double.Parse(splittedData[0]);

                    float key = float.Parse(splittedData[1]);
                    if (key < 0.0f)
                        key = 0.0f;

                    switch (dataStreamType)
                    {
                        case DataStreamType.Motion:
                            {
                                GetComponents<EventSender>()[1].SetEventValue("text_Text", data);
                                GetComponents<EventSender>()[1].Deploy();
                                break;
                            }
                        case DataStreamType.PerformanceMetrics:
                            {
                                GetComponents<EventSender>()[3].SetEventValue("text_Text", data);
                                GetComponents<EventSender>()[3].Deploy();
                                break;
                            }
                    }
                }

                private void UpdateDataRepresentationContainer(double engData, double excData, double lexData, double strData, double relData, double intData, double focData)
                {
                    _engLastData = engData;
                    _excLastData = excData;
                    _lexLastData = lexData;
                    _strLastData = strData;
                    _relLastData = relData;
                    _intLastData = intData;
                    _focLastData = focData;

                    GameObject barContainer1 = GameObject.Find("BarContainer");
                    if (barContainer1 != null)
                    {
                        GameObject bar = barContainer1.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (engData < 0.0f)
                            engData = 0.0f;

                        updateScaleValue.y = (float)engData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer1.transform.Find("BarHeadDescription").gameObject; //GameObject.Find("BarHeadDescription");
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (engData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer1.transform.Find("BarCaption").gameObject; //GameObject.Find("BarCaption");
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Engagement";
                    }

                    GameObject barContainer2 = GameObject.Find("BarContainer (1)");
                    if (barContainer2 != null)
                    {
                        GameObject bar = barContainer2.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (excData < 0.0f)
                            excData = 0.0f;

                        updateScaleValue.y = (float)excData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer2.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (excData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer2.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Excitement";
                    }

                    GameObject barContainer3 = GameObject.Find("BarContainer (2)");
                    if (barContainer3 != null)
                    {
                        GameObject bar = barContainer3.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (lexData < 0.0f)
                            lexData = 0.0f;

                        updateScaleValue.y = (float)lexData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer3.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (lexData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer3.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "L. term excitement";
                    }

                    GameObject barContainer4 = GameObject.Find("BarContainer (3)");
                    if (barContainer4 != null)
                    {
                        GameObject bar = barContainer4.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (strData < 0.0f)
                            strData = 0.0f;

                        updateScaleValue.y = (float)strData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer4.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (strData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer4.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Stress / Frustration";
                    }

                    GameObject barContainer5 = GameObject.Find("BarContainer (4)");
                    if (barContainer5 != null)
                    {
                        GameObject bar = barContainer5.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (relData < 0.0f)
                            relData = 0.0f;

                        updateScaleValue.y = (float)relData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer5.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (relData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer5.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Relaxation";
                    }

                    GameObject barContainer6 = GameObject.Find("BarContainer (5)");
                    if (barContainer6 != null)
                    {
                        GameObject bar = barContainer6.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (intData < 0.0f)
                            intData = 0.0f;

                        updateScaleValue.y = (float)intData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer6.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (intData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer6.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Interest / Affinity";
                    }

                    GameObject barContainer7 = GameObject.Find("BarContainer (6)");
                    if (barContainer7 != null)
                    {
                        GameObject bar = barContainer7.transform.Find("BarReprensentation").gameObject;
                        Vector3 updateScaleValue = bar.transform.localScale;

                        if (focData < 0.0f)
                            focData = 0.0f;

                        updateScaleValue.y = (float)focData * 10.0f;
                        bar.transform.localScale = updateScaleValue;

                        GameObject barHeadDescription = barContainer7.transform.Find("BarHeadDescription").gameObject;
                        TMPro.TextMeshPro barHeadDescriptionText = barHeadDescription.GetComponent<TextMeshPro>();
                        barHeadDescriptionText.text = "Current: " + (focData * 100).ToString("F2") + " %";

                        GameObject barCaption = barContainer7.transform.Find("BarCaption").gameObject;
                        TMPro.TextMeshPro barCaptionText = barCaption.GetComponent<TextMeshPro>();
                        barCaptionText.text = "Focus";
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
    }
}
