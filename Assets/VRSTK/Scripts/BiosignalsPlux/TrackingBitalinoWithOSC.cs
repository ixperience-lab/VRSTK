using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SharpOSC;
using System;
using System.Threading;

public class TrackingBitalinoWithOSC : MonoBehaviour
{
	Thread _test;

	[SerializeField]
	public string _remoteIP = "127.0.0.1"; //127.0.0.1 signifies a local host 
	[SerializeField]
	public int _sendToPort = 9000; //the port you will be sending from
	[SerializeField]
	public int _listenerPort = 5555; //the port you will be listening on

	[SerializeField]
	public string _address = "";

    [SerializeField]
    public string _timeStamp = "";

    [SerializeField]
	public int _sequenceNumber = -1;

	[SerializeField]
	public string _rawReceivedMessage;

	[SerializeField]
	public string _transferedReceivedMessage;

	[SerializeField]
	public int[] _sigitalIOs = new int[4];

	[SerializeField]
	public int[] _analogOutputs = new int[6];

	[SerializeField]
	public double[] _analogOutputsTransfered = new double[6];

	
	private Osc _handler;
	private UDPPacketIO _udp;

	private int _startTime;
	private int _endTime;

	// Use this for initialization
	void Start()
	{
		//Initializes on start up to listen for messages
		_udp = new UDPPacketIO();
		_udp.init(_remoteIP, _sendToPort, _listenerPort);
		_handler = new Osc();
		_handler.init(_udp);
		_handler.SetAllMessageHandler(AllMessageHandler);
		Debug.Log("OSC Connection initialized");

	}

	// Update is called once per frame
	void Update()
	{
	}

    void OnDisable()
    {
        if (_handler != null)
		    _handler.Cancel();
        if (_udp != null)
		    _udp.Close();
	}

    private void OnApplicationQuit()
    {
        if (_handler != null)
            _handler.Cancel();
        if (_udp != null)
            _udp.Close();
    }

    //These functions are called when messages are received
    //Access values via: oscMessage.Values[0], oscMessage.Values[1], etc
    public void AllMessageHandler(OscMessage oscMessage)
	{
		string msgString = Osc.OscMessageToString(oscMessage); //the message and value combined
		string msgAddress = oscMessage.Address; //the message address
		_rawReceivedMessage = msgString;
		Debug.Log(msgString); //log the message and values coming from OSC
		if (msgAddress.Equals(_address))
        {
            // message structure = "{0} - Seq[{1}] : O[{2} {3} {4} {5}] ; A[{6} {7} {8} {9} {10} {11}]"
            _timeStamp = msgString.Split(' ')[1];

            int startIndex = msgString.IndexOf("Seq[") + 4;
            for (int i = 0; i < msgString.Length; i++)
                if (msgString.Substring(startIndex + i, 1).Equals("]"))
                {
                    _sequenceNumber = Convert.ToInt32(msgString.Substring(startIndex, i));
                    break;
                }

            startIndex = msgString.IndexOf("O[") + 2;
            for (int i = 0; i < msgString.Length; i++)
                if (msgString.Substring(startIndex + i, 1).Equals("]"))
                {
                    _sigitalIOs[0] = msgString.Substring(startIndex, i).Split(' ')[0] == "True" ? 1 : 0;
                    _sigitalIOs[1] = msgString.Substring(startIndex, i).Split(' ')[1] == "True" ? 1 : 0;
                    _sigitalIOs[2] = msgString.Substring(startIndex, i).Split(' ')[2] == "True" ? 1 : 0;
                    _sigitalIOs[3] = msgString.Substring(startIndex, i).Split(' ')[3] == "True" ? 1 : 0;
                    break;
                }

            startIndex = msgString.IndexOf("A[") + 2;
            for (int i = 0; i < msgString.Length; i++)
                if (msgString.Substring(startIndex + i, 1).Equals("]"))
                {
                    // EDA (Electrodermal Activity) port A3
                    {
                        int eda_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[0]);
                        _analogOutputs[0] = eda_raw;
                        Debug.Log("A3 raw = " + eda_raw);
                        // Transfer function [0uS, 25uS] (micro Siemens)
                        int VCC = 3; // Operating voltage
                        int ADC = eda_raw; // Value sampled form the channel
                        int n = 10; // Number of bits of the channel
                        float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float)VCC) / 0.12f;
                        Debug.Log("A3 (micro Siemens) = " + EDA_uS);
                        Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
                        _analogOutputsTransfered[0] = EDA_uS;
                    }

                    // ECG (Electrpcardiography)	
                    {
                        int ecg_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[1]);
                        Debug.Log("A2 raw = " + ecg_raw);
                        _analogOutputs[1] = ecg_raw;
                        // Transfer function [-1.47mV, +1.47mV] (micro Volt)
                        {
                            int VCC = 3; // Operating voltage
                            int ADC = ecg_raw; // Value sampled form the channel
                            int n = 10; // Number of bits of the channel
                            int gECG = 1019; // sensor gain
                            float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float)VCC) / (float)gECG;
                            Debug.Log("A2 (Volt) = " + ECG_V);
                            Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
                            _analogOutputsTransfered[1] = ECG_V;
                        }
                    }

                    _analogOutputs[2] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[2]);
                    _analogOutputsTransfered[2] = _analogOutputs[2];
                    _analogOutputs[3] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[3]);
                    _analogOutputsTransfered[3] = _analogOutputs[3];
                    _analogOutputs[4] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[4]);
                    _analogOutputsTransfered[4] = _analogOutputs[4];
                    _analogOutputs[5] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[5]);
                    _analogOutputsTransfered[5] = _analogOutputs[5];
                    break;
                }

            _transferedReceivedMessage = string.Format("{0} {1} - Seq[{2}] : O[{3} {4} {5} {6}] ; A[{7} {8} {9} {10} {11} {12}]",
                msgAddress, _timeStamp, _sequenceNumber.ToString(), _sigitalIOs[0] == 1 ? "true":"false", _sigitalIOs[1] == 1 ? "true" : "false", _sigitalIOs[2] == 1 ? "true" : "false", _sigitalIOs[3] == 1 ? "true" : "false",
                _analogOutputsTransfered[0].ToString(), _analogOutputsTransfered[1].ToString(), _analogOutputsTransfered[2].ToString(), _analogOutputsTransfered[3].ToString(), _analogOutputsTransfered[4].ToString(), _analogOutputsTransfered[5].ToString());
        }
	}

}
