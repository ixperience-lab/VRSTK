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
	public int _sequenceNumber = -1;

	[SerializeField]
	public int[] _sigitalIOs = new int[4];

	[SerializeField]
	public int[] _analogOutputs = new int[6];

	[SerializeField]
	public double[] _analogOutputsTransfered = new double[6];

	[SerializeField]
	public string _rawReceivedMessage;

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
		//Debug.Log("delta time in ms: " + Time.deltaTime.ToString());
	}

    private void FixedUpdate()
    {
		//Debug.Log("fixed delta time in ms: " + Time.fixedDeltaTime.ToString());
	}

 //   void OnDisable()
	//{
	//	_handler.Cancel();
	//	_udp.Close();
	//}

    private void OnApplicationQuit()
    {
		_handler.Cancel();
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
            //_test = new Thread(TrasnferRawValues(msgString));
            //StartCoroutine(TrasnferRawValues(_rawReceivedMessage));
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
                    _analogOutputs[0] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[0]);
                    _analogOutputsTransfered[0] = _analogOutputs[0];

                    // ECG (Electrpcardiography)	
                    {
                        int ecg_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[1]);
                        Debug.Log("A2 raw = " + ecg_raw);
                        _analogOutputs[1] = ecg_raw;
                        // Transfer function [-1.47mV, +1.47mV] (micro Volt)
                        {
                            int VCC = 3; // Operating voltage
                            int ADC = ecg_raw; // Value sampled form the channel
                            int n = 8; // Number of bits of the channel
                            int gECG = 1019; // sensor gain
                            float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float)VCC) / (float)gECG;
                            Debug.Log("A2 (Volt) = " + ECG_V);
                            Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
                            _analogOutputsTransfered[1] = ECG_V;
                        }
                    }

                    // EDA (Electrodermal Activity) port A3
                    {
                        int eda_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[2]);
                        _analogOutputs[2] = eda_raw;
                        Debug.Log("A3 raw = " + eda_raw);
                        // Transfer function [0uS, 25uS] (micro Siemens)
                        int VCC = 3; // Operating voltage
                        int ADC = eda_raw; // Value sampled form the channel
                        int n = 8; // Number of bits of the channel
                        float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float)VCC) / 0.12f;
                        Debug.Log("A3 (micro Siemens) = " + EDA_uS);
                        Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
                        _analogOutputsTransfered[2] = EDA_uS;
                    }

                    _analogOutputs[3] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[3]);
                    _analogOutputsTransfered[3] = _analogOutputs[3];
                    _analogOutputs[4] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[4]);
                    _analogOutputsTransfered[4] = _analogOutputs[4];
                    _analogOutputs[5] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[5]);
                    _analogOutputsTransfered[5] = _analogOutputs[5];
                    break;
                }
        }
	}

	// Read Thread.  Loops waiting for packets.  When a packet is received, it is
	// dispatched to any waiting All Message Handler.  Also, the address is looked up and
	// any matching handler is called.
	private void TrasnferRawValues(string msgString)
	{
		try
		{
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
					_analogOutputs[0] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[0]);
					_analogOutputsTransfered[0] = _analogOutputs[0];

					// ECG (Electrpcardiography)	
					{
						int ecg_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[1]);
						Debug.Log("A2 raw = " + ecg_raw);
						_analogOutputs[1] = ecg_raw;
						// Transfer function [-1.47mV, +1.47mV] (micro Volt)
						{
							int VCC = 3; // Operating voltage
							int ADC = ecg_raw; // Value sampled form the channel
							int n = 8; // Number of bits of the channel
							int gECG = 1019; // sensor gain
							float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float)VCC) / (float)gECG;
							Debug.Log("A2 (Volt) = " + ECG_V);
							Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
							_analogOutputsTransfered[1] = ECG_V;
						}
					}

					// EDA (Electrodermal Activity) port A3
					{
						int eda_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[2]);
						_analogOutputs[2] = eda_raw;
						Debug.Log("A3 raw = " + eda_raw);
						// Transfer function [0uS, 25uS] (micro Siemens)
						int VCC = 3; // Operating voltage
						int ADC = eda_raw; // Value sampled form the channel
						int n = 8; // Number of bits of the channel
						float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float)VCC) / 0.12f;
						Debug.Log("A3 (micro Siemens) = " + EDA_uS);
						Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
						_analogOutputsTransfered[2] = EDA_uS;
					}

					_analogOutputs[3] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[3]);
					_analogOutputsTransfered[3] = _analogOutputs[3];
					_analogOutputs[4] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[4]);
					_analogOutputsTransfered[4] = _analogOutputs[4];
					_analogOutputs[5] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[5]);
					_analogOutputsTransfered[5] = _analogOutputs[5];
					break;
				}

			//while (ReaderRunning)
			//{
			//	byte[] buffer = new byte[1000];
			//	int length = OscPacketIO.ReceivePacket(buffer);
			//	//Debug.Log("received packed of len=" + length);
			//	if (length > 0)
			//	{
			//		ArrayList messages = Osc.PacketToOscMessages(buffer, length);
			//		foreach (OscMessage om in messages)
			//		{
			//			if (AllMessageHandler != null)
			//				AllMessageHandler(om);
			//			OscMessageHandler h = (OscMessageHandler)Hashtable.Synchronized(AddressTable)[om.Address];
			//			if (h != null)
			//				h(om);
			//		}
			//	}
			//	else
			//		Thread.Sleep(20);
			//}
		}
		catch (Exception e)
		{
			//Debug.Log("ThreadAbortException"+e);
		}
		finally
		{
			//Debug.Log("terminating thread - clearing handlers");
			//Cancel();
			//Hashtable.Synchronized(AddressTable).Clear();
		}

	}

	//IEnumerator TrasnferRawValues(string msgString)
	//{
	//	int startIndex = msgString.IndexOf("Seq[") + 4;
	//	for (int i = 0; i < msgString.Length; i++)
	//		if (msgString.Substring(startIndex + i, 1).Equals("]"))
	//		{
	//			_sequenceNumber = Convert.ToInt32(msgString.Substring(startIndex, i));
	//			break;
	//		}

	//	startIndex = msgString.IndexOf("O[") + 2;
	//	for (int i = 0; i < msgString.Length; i++)
	//		if (msgString.Substring(startIndex + i, 1).Equals("]"))
	//		{
	//			_sigitalIOs[0] = msgString.Substring(startIndex, i).Split(' ')[0] == "True" ? 1 : 0;
	//			_sigitalIOs[1] = msgString.Substring(startIndex, i).Split(' ')[1] == "True" ? 1 : 0;
	//			_sigitalIOs[2] = msgString.Substring(startIndex, i).Split(' ')[2] == "True" ? 1 : 0;
	//			_sigitalIOs[3] = msgString.Substring(startIndex, i).Split(' ')[3] == "True" ? 1 : 0;
	//			break;
	//		}

	//	startIndex = msgString.IndexOf("A[") + 2;
	//	for (int i = 0; i < msgString.Length; i++)
	//		if (msgString.Substring(startIndex + i, 1).Equals("]"))
	//		{
	//			_analogOutputs[0] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[0]);
	//			_analogOutputsTransfered[0] = _analogOutputs[0];

	//			// ECG (Electrpcardiography)	
	//			{
	//				int ecg_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[1]);
	//				Debug.Log("A2 raw = " + ecg_raw);
	//				_analogOutputs[1] = ecg_raw;
	//				// Transfer function [-1.47mV, +1.47mV] (micro Volt)
	//				{
	//					int VCC = 3; // Operating voltage
	//					int ADC = ecg_raw; // Value sampled form the channel
	//					int n = 8; // Number of bits of the channel
	//					int gECG = 1019; // sensor gain
	//					float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float)VCC) / (float)gECG;
	//					Debug.Log("A2 (Volt) = " + ECG_V);
	//					Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
	//					_analogOutputsTransfered[1] = ECG_V;
	//				}
	//			}

	//			// EDA (Electrodermal Activity) port A3
	//			{
	//				int eda_raw = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[2]);
	//				_analogOutputs[2] = eda_raw;
	//				Debug.Log("A3 raw = " + eda_raw);
	//				// Transfer function [0uS, 25uS] (micro Siemens)
	//				int VCC = 3; // Operating voltage
	//				int ADC = eda_raw; // Value sampled form the channel
	//				int n = 8; // Number of bits of the channel
	//				float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float)VCC) / 0.12f;
	//				Debug.Log("A3 (micro Siemens) = " + EDA_uS);
	//				Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
	//				_analogOutputsTransfered[2] = EDA_uS;
	//			}

	//			_analogOutputs[3] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[3]);
	//			_analogOutputsTransfered[3] = _analogOutputs[3];
	//			_analogOutputs[4] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[4]);
	//			_analogOutputsTransfered[4] = _analogOutputs[4];
	//			_analogOutputs[5] = Convert.ToInt32(msgString.Substring(startIndex, i).Split(' ')[5]);
	//			_analogOutputsTransfered[5] = _analogOutputs[5];
	//			break;
	//		}

	//	yield return null;
	//	//Color c = renderer.material.color;
	//	//for (float alpha = 1f; alpha >= 0; alpha -= 0.1f)
	//	//{
	//	//	c.a = alpha;
	//	//	renderer.material.color = c;
	//	//	yield return null;
	//	//}
	//}
}
