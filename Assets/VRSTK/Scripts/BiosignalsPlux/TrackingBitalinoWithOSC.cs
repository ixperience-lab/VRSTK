using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SharpOSC;
using System;

public class TrackingBitalinoWithOSC : MonoBehaviour
{
	[SerializeField]
	public string RemoteIP = "127.0.0.1"; //127.0.0.1 signifies a local host 
	[SerializeField]
	public int SendToPort = 9000; //the port you will be sending from
	[SerializeField]
	public int ListenerPort = 5555; //the port you will be listening on

	[SerializeField]
	public int SequenceNumber = -1;

	[SerializeField]
	public int[] DigitalIO = new int[4];

	[SerializeField]
	public int[] AnalogOutput = new int[6];

	[SerializeField]
	public double[] AnalogOutputTransfered = new double[6];

	private Osc handler;
	private UDPPacketIO udp;

	// Use this for initialization
	void Start()
	{
		//Initializes on start up to listen for messages
		udp = new UDPPacketIO();
		udp.init(RemoteIP, SendToPort, ListenerPort);
		handler = new Osc();
		handler.init(udp);
		handler.SetAllMessageHandler(AllMessageHandler);
		Debug.Log("OSC Connection initialized");

	}

	// Update is called once per frame
	void Update()
	{

	}

	void OnDisable()
	{
		udp.Close();
	}


	//These functions are called when messages are received
	//Access values via: oscMessage.Values[0], oscMessage.Values[1], etc

	public void AllMessageHandler(OscMessage oscMessage)
	{
		string msgString = Osc.OscMessageToString(oscMessage); //the message and value combined
		string msgAddress = oscMessage.Address; //the message address
		Debug.Log(msgString); //log the message and values coming from OSC
		if (msgAddress.Equals("/Bitalino/Frame"))
        {
			int startIndex = msgString.IndexOf("Seq[") + 4;
			for (int i = 0; i < msgString.Length; i++)
				if (msgString.Substring(startIndex + i, 1).Equals("]"))
				{
					SequenceNumber = Convert.ToInt32(msgString.Substring(startIndex, i));
					break;
				}

			startIndex = msgString.IndexOf("O[") + 2;
			for (int i = 0; i < msgString.Length; i++)
				if (msgString.Substring(startIndex + i, 1).Equals("]"))
				{
					DigitalIO[0] = msgString.Substring(startIndex, i).Split(' ')[0] == "True" ? 1 : 0;
					DigitalIO[1] = msgString.Substring(startIndex, i).Split(' ')[1] == "True" ? 1 : 0;
					DigitalIO[2] = msgString.Substring(startIndex, i).Split(' ')[2] == "True" ? 1 : 0;
					DigitalIO[3] = msgString.Substring(startIndex, i).Split(' ')[3] == "True" ? 1 : 0;
					break;
				}

			startIndex = msgString.IndexOf("A[") + 2;
			for (int i = 0; i < msgString.Length; i++)
				if (msgString.Substring(startIndex + i, 1).Equals("]"))
				{
					AnalogOutput[0] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[0]);
					AnalogOutputTransfered[0] = AnalogOutput[0];

					// ECG (Electrpcardiography)	
					{
						int ecg_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[1]);
						Debug.Log("A2 raw = " + ecg_raw);
						AnalogOutput[1] = ecg_raw;
						// Transfer function [-1.47mV, +1.47mV] (micro Volt)
						{
							int VCC = 3; // Operating voltage
							int ADC = ecg_raw; // Value sampled form the channel
							int n = 8; // Number of bits of the channel
							int gECG = 1019; // sensor gain
							float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float)VCC) / (float)gECG;
							Debug.Log("A2 (Volt) = " + ECG_V);
							Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
							AnalogOutputTransfered[1] = ECG_V;
						}
					}

					// EDA (Electrodermal Activity) port A3
					{
						int eda_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[2]);
						AnalogOutput[2] = eda_raw;
						Debug.Log("A3 raw = " + eda_raw);
						// Transfer function [0uS, 25uS] (micro Siemens)
						int VCC = 3; // Operating voltage
						int ADC = eda_raw; // Value sampled form the channel
						int n = 8; // Number of bits of the channel
						float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float)VCC) / 0.12f;
						Debug.Log("A3 (micro Siemens) = " + EDA_uS);
						Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
						AnalogOutputTransfered[2] = EDA_uS;
					}

				    AnalogOutput[3] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[3]);
					AnalogOutputTransfered[3] = AnalogOutput[3];
					AnalogOutput[4] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[4]);
					AnalogOutputTransfered[4] = AnalogOutput[4];
					AnalogOutput[5] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[5]);
					AnalogOutputTransfered[5] = AnalogOutput[5];
					break;
				}
		}
	}
}
