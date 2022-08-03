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

    // R-Peak detection by WH :-)
    //public double _rrInterval = 0f;
    //private List<double> _rTimeStampValues;
    //public float _rThreshold = 0.2f;

    //public double _rrInterval = 0f;
    //public int _heartRate = 0;
    public List<double> _rawValue;
    //public List<double> _valueTimeStamp;
    //public List<double> _rrCandidate;
    //public List<double> _rrTimeStampCandidate;

    // QRS detection scheme by Fraden and Neuman [5]
    // trying with 3 datapoint calculation
    //public float _amplitudeThreshold = 0.4f; // need to be max[X(n)]
    //public float _constantThreshold = 0.7f;
    //public double _y0;
    //public List<double> _y1;
    //public double _y2;
    //public List<double> _rTimeStampValues;
    //public List<double> _rrCandidate;
    //public List<double> _rrTimeStampCandidate;
    //public double _rrIntervalTime = 0f;

    // QRS detection sheme Gustafson [6]
    //public float _constantThreshold = 0.15f;
    //public double _y;
    //public List<double> _xValues;
    //public List<double> _rTimeStampValues;
    //public List<double> _rrCandidate;
    //public List<double> _rrTimeStampCandidate;

    //public List<float> _rawValue2;

    //public static int M = 5;
    //public static int N = 30;
    //public static int winSize = 90;
    //public static float HP_CONSTANT = (float)1 / M;

    private int _WINDOWSIZE = 20;   // Integrator window size, in samples. The article recommends 150ms. So, FS*0.15.
                                    // However, you should check empirically if the waveform looks ok.
    private int _NOSAMPLE = -32000; // An indicator that there are no more samples to read. Use an impossible value for a sample.
    private int _FS = 360;          // Sampling frequency.
    private int _BUFFSIZE = 1800;    // The size of the buffers (in samples). Must fit more than 1.66 times an RR interval, which
                                    // typically could be around 1 second.

    private int _DELAY = 22;		// Delay introduced by the filters. Filter only output samples after this one.
                                    // Set to 0 if you want to keep the delay. Fixing the delay results in DELAY less samples
                                    // in the final end result.

    // Use this for initialization
    void Start()
	{
        _rawValue = new List<double>();
        //_rawValue2 = new List<float>();
        //_valueTimeStamp = new List<double>();
        //_rrTimeStampCandidate = new List<double>();
        //_rrCandidate = new List<double>();
        //_rTimeStampValues = new List<double>();
        //Initializes on start up to listen for messages

        //_y1 = new List<double>();
        //_rTimeStampValues = new List<double>();

        //_xValues = new List<double>();
        //_rTimeStampValues = new List<double>();
        //_rrCandidate = new List<double>();
        //_rrTimeStampCandidate = new List<double>();


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
                            float VCC = 3.3f; // Operating voltage
                            int ADC = ecg_raw; // Value sampled form the channel
                            int n = 10; // Number of bits of the channel
                            int gECG = 1100; // sensor gain
                            float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * VCC) / (float)gECG;
                            float ECG_mV = (ECG_V * 1000);

                            //_rawValue.Add(ecg_raw);
                            ////_rawValue2.Add(ecg_raw);
                            //_valueTimeStamp.Add(double.Parse(_timeStamp));

                            //if (_rawValue.Count > _BUFFSIZE - 2)
                            //{
                            //    _rawValue.Add(_NOSAMPLE);
                            //    _rawValue.Add(_NOSAMPLE);

                            //    //SsfSegmenter(FilterSignal(_rawValue2.ToArray()));
                            //    // _rawValue2.Clear();

                            //    // detect();
                            //    PanTompkins();
                            //    _rawValue.Clear();
                            //    _valueTimeStamp.Clear();
                            //}

                            //if (ECG_mV >= _rThreshold)
                            //    _rTimeStampValues.Add(double.Parse(_timeStamp));

                            //if (_rrCandidate.Count > 1)
                            //{
                            //    for (int j = 1; j < _rrCandidate.Count; j++)
                            //    {
                            //        _rrInterval = Mathf.Abs((float)(_rrCandidate[j] - _rrCandidate[j-1]));
                            //        if (_rrInterval > 0f && (int)(_FS * (60f / _rrInterval)) > 40 && (int)(_FS * (60f / _rrInterval)) < 200)
                            //            _heartRate = (int)(_FS * (60f / _rrInterval));
                            //    }
                            //    _rrCandidate.Clear();
                            //    _rrTimeStampCandidate.Clear();
                            //}

                            //if (_rrCandidate.Count > 1)
                            //{
                            //    for (int j = 1; j < _rrCandidate.Count; j++)
                            //    {
                            //        _rrInterval += Mathf.Abs((float)(_rrCandidate[j] - _rrCandidate[j - 1]));
                            //    }

                            //    _rrInterval /= _rrCandidate.Count;

                            //    if (_rrInterval > 0f)//&& (int)(60f / _rrInterval) > 40 && (int)(60f / _rrInterval) < 200)
                            //        _heartRate = (int)(6000f / _rrInterval);

                            //    _rrTimeStampCandidate.Clear();
                            //    _rrCandidate.Clear();
                            //}

                            //if (_rrCandidate.Count > 1)
                            //{
                            //    _rrInterval = Mathf.Abs((float)(_rrCandidate[1] - _rrCandidate[0]));
                            //    if (_rrInterval > 0f)
                            //        _heartRate = (int)(6000f / _rrInterval);//(int)(_FS * (60f / _rrInterval));
                            //    _rrCandidate.Clear();
                            //    _rrTimeStampCandidate.Clear();
                            //}

                            //if (_rrTimeStampCandidate.Count > 1)
                            //{
                            //    for (int j = 1; j < _rrTimeStampCandidate.Count; j++)
                            //    {
                            //        _rrInterval += Mathf.Abs((float)(_rrTimeStampCandidate[j] - _rrTimeStampCandidate[j - 1]));
                            //    }

                            //    _rrInterval /= _rrTimeStampCandidate.Count;

                            //    if (_rrInterval > 0f)//&& (int)(60f / _rrInterval) > 40 && (int)(60f / _rrInterval) < 200)
                            //        _heartRate = (int)(6000f / _rrInterval);

                            //    _rrTimeStampCandidate.Clear();
                            //    _rrCandidate.Clear();
                            //}

                            //if (_rrTimeStampCandidate.Count > 1)
                            //{
                            //    _rrInterval = _rrTimeStampCandidate[1] - _rrTimeStampCandidate[0];
                            //    if (_rrInterval > 0f)
                            //        _heartRate = (int)(6000f / _rrInterval);

                            //    _rrCandidate.Clear();
                            //    _rrTimeStampCandidate.Clear();
                            //}

                            Debug.Log("A2 (Volt) = " + ECG_V);
                            Debug.Log("A2 (milli Volt) = " + ECG_mV);
                            _analogOutputsTransfered[1] = ECG_mV;
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

    void PanTompkins()
    {
        // The signal array is where the most recent samples are kept. The other arrays are the outputs of each
        // filtering module: DC Block, low pass, high pass, integral etc.
        // The output is a buffer where we can change a previous result (using a back search) before outputting.
        double[] signal = new double[_BUFFSIZE], dcblock = new double[_BUFFSIZE], lowpass = new double[_BUFFSIZE], highpass = new double[_BUFFSIZE];
        double[] derivative = new double[_BUFFSIZE], squared = new double[_BUFFSIZE], integral = new double[_BUFFSIZE], outputSignal = new double[_BUFFSIZE];

        // rr1 holds the last 8 RR intervals. rr2 holds the last 8 RR intervals between rrlow and rrhigh.
        // rravg1 is the rr1 average, rr2 is the rravg2. rrlow = 0.92*rravg2, rrhigh = 1.08*rravg2 and rrmiss = 1.16*rravg2.
        // rrlow is the lowest RR-interval considered normal for the current heart beat, while rrhigh is the highest.
        // rrmiss is the longest that it would be expected until a new QRS is detected. If none is detected for such
        // a long interval, the thresholds must be adjusted.
        int [] rr1 = new int[8], rr2 = new int[8];
        int rravg1, rravg2 = 0, rrlow = 0, rrhigh = 0, rrmiss = 0;

        // i and j are iterators for loops.
        // sample counts how many samples have been read so far.
        // lastQRS stores which was the last sample read when the last R sample was triggered.
        // lastSlope stores the value of the squared slope when the last R sample was triggered.
        // currentSlope helps calculate the max. square slope for the present sample.
        // These are all long unsigned int so that very long signals can be read without messing the count.
        uint i, j, sample = 0, lastQRS = 0, lastSlope = 0, currentSlope = 0;

        // This variable is used as an index to work with the signal buffers. If the buffers still aren't
        // completely filled, it shows the last filled position. Once the buffers are full, it'll always
        // show the last position, and new samples will make the buffers shift, discarding the oldest
        // sample and storing the newest one on the last position.
        int current;

        // There are the variables from the original Pan-Tompkins algorithm.
        // The ones ending in _i correspond to values from the integrator.
        // The ones ending in _f correspond to values from the DC-block/low-pass/high-pass filtered signal.
        // The peak variables are peak candidates: signal values above the thresholds.
        // The threshold 1 variables are the threshold variables. If a signal sample is higher than this threshold, it's a peak.
        // The threshold 2 variables are half the threshold 1 ones. They're used for a back search when no peak is detected for too long.
        // The spk and npk variables are, respectively, running estimates of signal and noise peaks.
        double peak_i = 0, peak_f = 0, threshold_i1 = 0, threshold_i2 = 0, threshold_f1 = 0, threshold_f2 = 0, spk_i = 0, spk_f = 0, npk_i = 0, npk_f = 0;

        // qrs tells whether there was a detection or not.
        // regular tells whether the heart pace is regular or not.
        // prevRegular tells whether the heart beat was regular before the newest RR-interval was calculated.
        bool qrs, regular = true, prevRegular;

        // Initializing the RR averages
        for (i = 0; i < 8; i++)
        {
            rr1[i] = 0;
            rr2[i] = 0;
        }

        try
        {
            // The main loop where everything proposed in the paper happens. Ends when there are no more signal samples.
            do
            {
                // Test if the buffers are full.
                // If they are, shift them, discarding the oldest sample and adding the new one at the end.
                // Else, just put the newest sample in the next free position.
                // Update 'current' so that the program knows where's the newest sample.
                //if (sample >= _BUFFSIZE)
                //{
                //    for (i = 0; i < _BUFFSIZE - 1; i++)
                //    {
                //        signal[i] = signal[i + 1];
                //        dcblock[i] = dcblock[i + 1];
                //        lowpass[i] = lowpass[i + 1];
                //        highpass[i] = highpass[i + 1];
                //        derivative[i] = derivative[i + 1];
                //        squared[i] = squared[i + 1];
                //        integral[i] = integral[i + 1];
                //        outputSignal[i] = outputSignal[i + 1];
                //    }
                //    current = _BUFFSIZE - 1;
                //}
                //else
                {
                    current = (int)sample;
                }
                signal[current] = _rawValue[current];

                // If no sample was read, stop processing!
                //if (signal[current] == NOSAMPLE)
                //    break;
                sample++; // Update sample counter

                // DC Block filter
                // This was not proposed on the original paper.
                // It is not necessary and can be removed if your sensor or database has no DC noise.
                if (current >= 1)
                    dcblock[current] = signal[current] - signal[current - 1] + 0.995 * dcblock[current - 1];
                else
                    dcblock[current] = 0;

                // Low Pass filter
                // Implemented as proposed by the original paper.
                // y(nT) = 2y(nT - T) - y(nT - 2T) + x(nT) - 2x(nT - 6T) + x(nT - 12T)
                // Can be removed if your signal was previously filtered, or replaced by a different filter.
                lowpass[current] = dcblock[current];
                if (current >= 1)
                    lowpass[current] += 2 * lowpass[current - 1];
                if (current >= 2)
                    lowpass[current] -= lowpass[current - 2];
                if (current >= 6)
                    lowpass[current] -= 2 * dcblock[current - 6];
                if (current >= 12)
                    lowpass[current] += dcblock[current - 12];

                // High Pass filter
                // Implemented as proposed by the original paper.
                // y(nT) = 32x(nT - 16T) - [y(nT - T) + x(nT) - x(nT - 32T)]
                // Can be removed if your signal was previously filtered, or replaced by a different filter.
                highpass[current] = -lowpass[current];
                if (current >= 1)
                    highpass[current] -= highpass[current - 1];
                if (current >= 16)
                    highpass[current] += 32 * lowpass[current - 16];
                if (current >= 32)
                    highpass[current] += lowpass[current - 32];

                // Derivative filter
                // This is an alternative implementation, the central difference method.
                // f'(a) = [f(a+h) - f(a-h)]/2h
                // The original formula used by Pan-Tompkins was:
                // y(nT) = (1/8T)[-x(nT - 2T) - 2x(nT - T) + 2x(nT + T) + x(nT + 2T)]
                derivative[current] = highpass[current];
                if (current > 0)
                    derivative[current] -= highpass[current - 1];

                // This just squares the derivative, to get rid of negative values and emphasize high frequencies.
                // y(nT) = [x(nT)]^2.
                squared[current] = derivative[current] * derivative[current];

                // Moving-Window Integration
                // Implemented as proposed by the original paper.
                // y(nT) = (1/N)[x(nT - (N - 1)T) + x(nT - (N - 2)T) + ... x(nT)]
                // WINDOWSIZE, in samples, must be defined so that the window is ~150ms.

                integral[current] = 0;
                for (i = 0; i < _WINDOWSIZE; i++)
                {
                    if (current >= (double)i)
                        integral[current] += squared[current - i];
                    else
                        break;
                }
                integral[current] /= (double)i;

                qrs = false;

                // If the current signal is above one of the thresholds (integral or filtered signal), it's a peak candidate.
                if (integral[current] >= threshold_i1 || highpass[current] >= threshold_f1)
                {
                    peak_i = integral[current];
                    peak_f = highpass[current];
                }

                // If both the integral and the signal are above their thresholds, they're probably signal peaks.
                if ((integral[current] >= threshold_i1) && (highpass[current] >= threshold_f1))
                {
                    // There's a 200ms latency. If the new peak respects this condition, we can keep testing.
                    if (sample > lastQRS + _FS / 5) // + 72)
                    {
                        // If it respects the 200ms latency, but it doesn't respect the 360ms latency, we check the slope.
                        if (sample <= lastQRS + (uint)(0.36 * _FS)) //+ (uint)(0.1296 * _FS))
                        {
                            // The squared slope is "M" shaped. So we have to check nearby samples to make sure we're really looking
                            // at its peak value, rather than a low one.
                            currentSlope = 0;
                            for (j = (uint)current - 10; j <= current; j++)
                                if (squared[j] > currentSlope)
                                    currentSlope = (uint)squared[j];

                            if (currentSlope <= (double)(lastSlope / 2))
                            {
                                qrs = false;
                            }
                            else
                            {
                                spk_i = 0.125 * peak_i + 0.875 * spk_i;
                                threshold_i1 = npk_i + 0.25 * (spk_i - npk_i);
                                threshold_i2 = 0.5 * threshold_i1;

                                spk_f = 0.125 * peak_f + 0.875 * spk_f;
                                threshold_f1 = npk_f + 0.25 * (spk_f - npk_f);
                                threshold_f2 = 0.5 * threshold_f1;

                                lastSlope = currentSlope;
                                qrs = true;
                                //_rrCandidate.Add(spk_f);
                            }
                        }
                        // If it was above both thresholds and respects both latency periods, it certainly is a R peak.
                        else
                        {
                            currentSlope = 0;
                            for (j = (uint)current - 10; j <= current; j++)
                                if (squared[j] > currentSlope)
                                    currentSlope = (uint)squared[j];

                            spk_i = 0.125 * peak_i + 0.875 * spk_i;
                            threshold_i1 = npk_i + 0.25 * (spk_i - npk_i);
                            threshold_i2 = 0.5 * threshold_i1;

                            spk_f = 0.125 * peak_f + 0.875 * spk_f;
                            threshold_f1 = npk_f + 0.25 * (spk_f - npk_f);
                            threshold_f2 = 0.5 * threshold_f1;

                            lastSlope = currentSlope;
                            qrs = true;

                            //_rrCandidate.Add(spk_f);
                        }
                    }
                    // If the new peak doesn't respect the 200ms latency, it's noise. Update thresholds and move on to the next sample.
                    else
                    {
                        peak_i = integral[current];
                        npk_i = 0.125 * peak_i + 0.875 * npk_i;
                        threshold_i1 = npk_i + 0.25 * (spk_i - npk_i);
                        threshold_i2 = 0.5 * threshold_i1;
                        peak_f = highpass[current];
                        npk_f = 0.125 * peak_f + 0.875 * npk_f;
                        threshold_f1 = npk_f + 0.25 * (spk_f - npk_f);
                        threshold_f2 = 0.5 * threshold_f1;
                        qrs = false;
                        outputSignal[current] = qrs ? 1f : 0f;
                        if (sample > _DELAY + _BUFFSIZE)
                            Debug.Log("outputSignal" + outputSignal[0]);
                        continue;
                    }

                }

                // If a R-peak was detected, the RR-averages must be updated.
                if (qrs)
                {
                    // Add the newest RR-interval to the buffer and get the new average.
                    rravg1 = 0;
                    for (i = 0; i < 7; i++)
                    {
                        rr1[i] = rr1[i + 1];
                        rravg1 += rr1[i];
                    }
                    rr1[7] = (int)(sample - lastQRS);
                    lastQRS = sample;
                    rravg1 += rr1[7];
                    rravg1 = (int)((double)rravg1 * 0.125);

                    // If the newly-discovered RR-average is normal, add it to the "normal" buffer and get the new "normal" average.
                    // Update the "normal" beat parameters.
                    if ((rr1[7] >= rrlow) && (rr1[7] <= rrhigh))
                    {
                        rravg2 = 0;
                        for (i = 0; i < 7; i++)
                        {
                            rr2[i] = rr2[i + 1];
                            rravg2 += rr2[i];
                        }
                        rr2[7] = rr1[7];
                        rravg2 += rr2[7];
                        rravg2 = (int)((double)rravg2 * 0.125);
                        rrlow = (int)(0.92 * rravg2);
                        rrhigh = (int)(1.16 * rravg2);
                        rrmiss = (int)(1.66 * rravg2);
                    }

                    prevRegular = regular;
                    if (rravg1 == rravg2)
                    {
                        regular = true;
                    }
                    // If the beat had been normal but turned odd, change the thresholds.
                    else
                    {
                        regular = false;
                        if (prevRegular)
                        {
                            threshold_i1 /= 2;
                            threshold_f1 /= 2;
                        }
                    }
                }
                // If no R-peak was detected, it's important to check how long it's been since the last detection.
                else
                {
                    // If no R-peak was detected for too long, use the lighter thresholds and do a back search.
                    // However, the back search must respect the 200ms limit and the 360ms one (check the slope).
                    if ((sample - lastQRS > (uint)rrmiss) && (sample > lastQRS + _FS / 5)) // + 72))
                    {
                        for (i = (uint)current - (sample - lastQRS) + (uint) _FS / 5; i < (uint)current; i++) //+ 72; i < (uint)current; i++)
                        {
                            if ((integral[i] > threshold_i2) && (highpass[i] > threshold_f2))
                            {
                                currentSlope = 0;
                                for (j = i - 10; j <= i; j++)
                                    if (squared[j] > currentSlope)
                                        currentSlope = (uint)squared[j];

                                if ((currentSlope < (double)(lastSlope / 2)) && (i + sample) < lastQRS + 0.36 * lastQRS)
                                {
                                    qrs = false;
                                }
                                else
                                {
                                    peak_i = integral[i];
                                    peak_f = highpass[i];
                                    spk_i = 0.25 * peak_i + 0.75 * spk_i;
                                    spk_f = 0.25 * peak_f + 0.75 * spk_f;
                                    threshold_i1 = npk_i + 0.25 * (spk_i - npk_i);
                                    threshold_i2 = 0.5 * threshold_i1;
                                    lastSlope = currentSlope;
                                    threshold_f1 = npk_f + 0.25 * (spk_f - npk_f);
                                    threshold_f2 = 0.5 * threshold_f1;
                                    // If a signal peak was detected on the back search, the RR attributes must be updated.
                                    // This is the same thing done when a peak is detected on the first try.
                                    //RR Average 1
                                    rravg1 = 0;
                                    for (j = 0; j < 7; j++)
                                    {
                                        rr1[j] = rr1[j + 1];
                                        rravg1 += rr1[j];
                                    }
                                    rr1[7] = (int)(sample - ((uint)current - i) - lastQRS);
                                    qrs = true;
                                    lastQRS = sample - ((uint)current - i);
                                    rravg1 += rr1[7];
                                    rravg1 = (int)((double)rravg1 * 0.125);

                                    //RR Average 2
                                    if ((rr1[7] >= rrlow) && (rr1[7] <= rrhigh))
                                    {
                                        rravg2 = 0;
                                        for (i = 0; i < 7; i++)
                                        {
                                            rr2[i] = rr2[i + 1];
                                            rravg2 += rr2[i];
                                        }
                                        rr2[7] = rr1[7];
                                        rravg2 += rr2[7];
                                        rravg2 = (int)((double)rravg2 * 0.125);
                                        rrlow = (int)(0.92 * rravg2);
                                        rrhigh = (int)(1.16 * rravg2);
                                        rrmiss = (int)(1.66 * rravg2);
                                    }

                                    prevRegular = regular;
                                    if (rravg1 == rravg2)
                                    {
                                        regular = true;
                                    }
                                    else
                                    {
                                        regular = false;
                                        if (prevRegular)
                                        {
                                            threshold_i1 /= 2;
                                            threshold_f1 /= 2;
                                        }
                                    }

                                    //_rrCandidate.Add(spk_f);

                                    break;
                                }
                            }
                        }

                        if (qrs)
                        {
                            outputSignal[current] = 0;//false;
                            outputSignal[i] = 1;//true;
                            if (sample > _DELAY + _BUFFSIZE)
                                Debug.Log("outputSignal" + outputSignal[0]);
                            continue;
                        }
                    }

                    // Definitely no signal peak was detected.
                    if (!qrs)
                    {
                        // If some kind of peak had been detected, then it's certainly a noise peak. Thresholds must be updated accordinly.
                        if ((integral[current] >= threshold_i1) || (highpass[current] >= threshold_f1))
                        {
                            peak_i = integral[current];
                            npk_i = 0.125 * peak_i + 0.875 * npk_i;
                            threshold_i1 = npk_i + 0.25 * (spk_i - npk_i);
                            threshold_i2 = 0.5 * threshold_i1;
                            peak_f = highpass[current];
                            npk_f = 0.125 * peak_f + 0.875 * npk_f;
                            threshold_f1 = npk_f + 0.25 * (spk_f - npk_f);
                            threshold_f2 = 0.5 * threshold_f1;
                        }
                    }
                }
                // The current implementation outputs '0' for every sample where no peak was detected,
                // and '1' for every sample where a peak was detected. It should be changed to fit
                // the desired application.
                // The 'if' accounts for the delay introduced by the filters: we only start outputting after the delay.
                // However, it updates a few samples back from the buffer. The reason is that if we update the detection
                // for the current sample, we might miss a peak that could've been found later by backsearching using
                // lighter thresholds. The final waveform output does match the original signal, though.
                outputSignal[current] = qrs ? 1 : 0;
                if (sample > _DELAY + _BUFFSIZE)
                    Debug.Log("outputSignal: " + outputSignal[0]);
            } while (signal[current] != _NOSAMPLE);
        }
        catch(Exception e)
        {
            return;
        }
        // Output the last remaining samples on the buffer
        //for (i = 1; i < _BUFFSIZE; i++)
        //{
        //    if (outputSignal[i] == 1)
        //    {
        //        _rrTimeStampCandidate.Add(i);
        //        _rrCandidate.Add(_valueTimeStamp[(int)i]);
        //    }
        //    //Debug.Log("outputSignal" + outputSignal[i]);
        //}

        // These last two lines must be deleted if you are not working with files.
        //fclose(fin);
        //fclose(fout);
    }

    public void SsfSegmenter(float [] signal, float sampling_rate= 1000.0f, float threshold = 20f, float before = 0.03f, float after= 0.01f)
    {
        //"""ECG R-peak segmentation based on the Slope Sum Function (SSF).

        //Parameters
        //----------
        //signal : array
        //    Input filtered ECG signal.
        //sampling_rate : int, float, optional
        //    Sampling frequency(Hz).
        //threshold : float, optional
        //    SSF threshold.
        //before : float, optional
        //    Search window size before R-peak candidate(seconds).
        //after : float, optional
        //    Search window size after R-peak candidate(seconds).

        //Returns
        //-------
        //rpeaks : array
        //    R-peak location indices.

        //"""
    }  

}
