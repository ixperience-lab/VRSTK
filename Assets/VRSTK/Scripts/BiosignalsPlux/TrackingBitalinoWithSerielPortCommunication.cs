using System;
using System.Collections;
using System.Collections.Generic;
using System.IO.Ports;
using System.Text;
using UnityEngine;

public class TrackingBitalinoWithSerielPortCommunication : MonoBehaviour
{
    // Supported channel number codes:
    //  {1 channel - 0x01, 2 channels - 0x03, 3 channels - 0x07
    //  4 channels - 0x0F, 5 channels - 0x1F, 6 channels - 0x3F
    //  7 channels - 0x7F, 8 channels - 0xFF}
    
    // Maximum acquisition frequencies for number of channels:
    //  1 channel - 8000, 2 channels - 5000, 3 channels - 4000
    //  4 channels - 3000, 5 channels - 3000, 6 channels - 2000
    //  7 channels - 2000, 8 channels - 2000


    bool _battaryMessageSended = false;
    bool _versionInfSended = false;
    bool _started = false;
    bool _trigger = false;
    bool _readed = false;

    public SerialPort _serialPortStream;
    public string _port = "COM3";               // communication port
    public int _speed = 115200;                 // speed of the communication (baud)
    public string _version;                     // current version
    public int _currentBatteryThreshold = 30;   // current Battery threshold

    public bool[] _analogChannels = new bool[] { false, false, true, true, false, false }; // predefined analog channels
    public int[] _analogChannelsResults = new int[] { 0, 0, 0, 0, 0, 0 }; // predefined analog channels
    public int _samplingRate = 1000;

    public int[] _digitalOutputArray = new int[] { 0, 0, 1, 1 };

    public int _samples = 10;

    // Start is called before the first frame update
    void Start()
    {
        OpenConnection();
    }

    // Update is called once per frame
    void Update()
    {
        if (_serialPortStream == null || !_serialPortStream.IsOpen)
            return;

        if (Input.anyKey && _serialPortStream.IsOpen)
        {
            StopStream();
            CloseConnection();
        }


        //if (!_trigger && _readed)
        //{
        //    Trigger(_digitalArray);
        //    _readed = false;
        //}
        if (_started && _serialPortStream.IsOpen)
        {
            Read(_samples);
        }
        if(_versionInfSended && !_started)
        {
            StartStream(_samplingRate, _analogChannels);
            _started = true;
        }
        if (_battaryMessageSended && !_versionInfSended)
        {
            _version = Version();
            _versionInfSended = true;
        }
        if (_serialPortStream.IsOpen && !_battaryMessageSended)
        {
            BatteryCommand(_currentBatteryThreshold);
            _battaryMessageSended = true;
        }
    }

    //open bluetooth communication 
    public void OpenConnection()
    {
        if (_serialPortStream == null)
        {
            _serialPortStream = new SerialPort(_port, _speed, Parity.None, 8, StopBits.One);
        }
        if (_serialPortStream != null)
        {
            if (!_serialPortStream.IsOpen)
            {
                try
                {
                    _serialPortStream.ReadTimeout = 5000;  // sets the timeout value before reporting error
                    _serialPortStream.WriteTimeout = 5000;
                    _serialPortStream.RtsEnable = true;
                    
                    _serialPortStream.Open();  // opens the connection

                    Debug.Log("Port Open!");
                }
                catch (System.Exception e)
                {
                    Debug.Log(e.ToString());
                }
            }
        }
    }

    //close connexion
    public void CloseConnection()
    {
        if (_serialPortStream != null && _serialPortStream.IsOpen)
        {
            try
            {
                _serialPortStream.Close();
            }
            catch (System.Exception e)
            {
                Debug.Log(e.ToString());
            }
        }
    }

    // Possible values for parameter* value*:
    //    ===============  =======  =====================
    //    Range* value*  Corresponding threshold(Volts)
    //    ===============  =======  =====================
    //    Minimum* value*  0        3.4 Volts
    //    Maximum* value*  63       3.8 Volts
    //    ===============  =======  =====================
    private void BatteryCommand(int currentBatteryThreshold)
    {
        // <bat   threshold> 0  0 - Set battery threshold
        int batteryCommand = 0;
        if (0 <= currentBatteryThreshold && currentBatteryThreshold <= 63)
            batteryCommand = currentBatteryThreshold << 2;

        if (batteryCommand != 0) 
            _serialPortStream.Write(BitConverter.GetBytes(batteryCommand), 0, 4);

        //string receivedBufferEncoded = string.Empty;
        //int bufferSizeToRead = _serialPortStream.ReadBufferSize;
        //byte[] buffer = new byte[bufferSizeToRead];
        //{
        //    _serialPortStream.Read(buffer, 0, bufferSizeToRead);
        //    receivedBufferEncoded = Encoding.ASCII.GetString(buffer, 0, bufferSizeToRead);
        //    Debug.Log(receivedBufferEncoded);
        //}
    }


    // samplingRate: sampling frequency(Hz)
    //  type samplingRate: int    
    // analogChannels: channels to be acquired
    //  type analogChannels: array or list of int
    // Sets the sampling rate and starts acquisition in the analog channels set.
    // Setting the sampling rate and starting the acquisition implies the use of the method :meth:`send`.
    // Possible values for parameter* samplingRate*:
    //    * 1
    //    * 10
    //    * 100
    //    * 1000
    // Possible values, types, configurations and examples for parameter* analogChannels*:
    //    ===============  ====================================
    //    Values           0, 1, 2, 3, 4, 5
    //    Types            array []
    //    Configurations Any number of channels, identified by their value
    //    Examples         [0, 3, 4]
    //    ===============  ====================================
    private void StartStream(int samplingRate = 1000, bool[] channels = null)
    {
        //Sleep(150); 
        System.Threading.Thread.Sleep(150);

        if (channels is null)
        {
            throw new ArgumentNullException(nameof(channels));
        }

        int channelsLength = 0;

        byte[] samplingRateCommandInByte = new byte[1] { 0x00 };
        int samplingRateCommand = 0;
        switch (samplingRate)
        {
            case 1:
                samplingRateCommandInByte[0] = 0x03;
                samplingRateCommand = 0x03;
                break;
            case 10:
                samplingRateCommandInByte[0] = 0x43;
                samplingRateCommand = 0x43;
                break;
            case 100:
                samplingRateCommandInByte[0] = 0x83;
                samplingRateCommand = 0x83;
                break;
            case 1000:
                samplingRateCommandInByte[0] = 0xC3;
                samplingRateCommand = 0xC3;
                break;
        }

        //byte[] channelsMaskInByte = new byte[1] { 0x3F };
        int channelsMask;
        {
            // 0  0  A6 A5 A4 A3 A2 A1 - channelsMask -> all channels selected
            channelsMask = 0x3F;    // all 6 analog channels
            channelsLength = 6;
        }
        
        //  <Fs>  0  0  0  0  1  1 - Set sampling rate
        _serialPortStream.Write(samplingRateCommandInByte, 0, 1);//BitConverter.GetBytes(samplingRateCommand), 0, 4);

        System.Threading.Thread.Sleep(150);

        // A6 A5 A4 A3 A2 A1 0  1 - Start live mode with analog channel selection
        // A6 A5 A4 A3 A2 A1 1  0 - Start simulated mode with analog channel selection
        _serialPortStream.Write(BitConverter.GetBytes(((channelsMask << 2) | 0x01)), 0, 1);


        // python impl ...
        ////samplingRate == 1
        //int samplingRateCommand = 0;

        //if (samplingRate == 1000)
        //    samplingRateCommand = 3;
        //else if (samplingRate == 100)
        //    samplingRateCommand = 2;
        //else if (samplingRate == 10)
        //    samplingRateCommand = 1;

        //int samplingRateCommandTemp = ((samplingRateCommand << 6) | 0x03);
        //_serialPortStream.Write(BitConverter.GetBytes(samplingRateCommandTemp), 0, 4);

        //int startCommand = 1;
        //for (int i = 0; i < analogChannels.Length; i++)
        //    startCommand = startCommand | 1 << (2 + i);

        //_serialPortStream.Write(BitConverter.GetBytes(samplingRateCommandTemp), 0, 4);
        //_serialPortStream.Write(BitConverter.GetBytes(startCommand), 0, 4);

        //string receivedBufferEncoded = string.Empty;
        //int bufferSizeToRead = _serialPortStream.ReadBufferSize;
        //byte[] buffer = new byte[bufferSizeToRead];
        //{
        //    _serialPortStream.Read(buffer, 0, bufferSizeToRead);
        //    receivedBufferEncoded = Encoding.ASCII.GetString(buffer, 0, bufferSizeToRead);
        //    Debug.Log(receivedBufferEncoded);
        //}
    }

    private void StopStream()
    {
        // 0  0  0  0  0  0  0  0 - Go to idle mode
        int stopCommand = 0x00; 
        _serialPortStream.Write(BitConverter.GetBytes(stopCommand), 0, 4);
    }

    // digitalArray: array which acts on digital outputs according to the value: 0 or 1
    //  type digitalArray: array, tuple or list of int
    // Acts on digital output channels of the BITalino device.Triggering these digital outputs implies the use of the method :meth:`send`.
    // Each position of the array* digitalArray* corresponds to a digital output, in ascending order.Possible values, types, configurations and examples for parameter* digitalArray*:
    //    ===============  ====================================
    //    Values           0 or 1
    //    Types            array []
    //    Configurations   4 values, one for each digital channel output
    //    Examples         [1, 0, 1, 0] : Digital 0 and 2 will be set to 1 while Digital 1 and 3 will be set to 0
    //    ===============  ====================================       
    private void Trigger(int[] digitalArray) // [0, 0, 0, 0]
    {
        // python impl ...
        //int data = 3;
        //for (int i = 0; i < digitalArray.Length; i++) 
        //    data = (data | digitalArray[i] << (2 + i));

        //_serialPortStream.Write(BitConverter.GetBytes(data), 0, 4);

        int triggerCommand = 0x03;   // 0  0  O4 O3 O2 O1 1  1 - Set digital outputs

        for (int i = 0; i < digitalArray.Length; i++)
            if (digitalArray[i] == 1)
                triggerCommand |= (0x04 << i);

        _serialPortStream.Write(BitConverter.GetBytes(triggerCommand), 0, 4);
    }

    // samples: number of samples to acquire
    //  type nSamples: int
    // returns: array with the acquired data 
    // Acquires "samples" from BITalino.Reading samples from BITalino implies the use of the method "receive".
    // Requiring a low number of samples (e.g. samples = 1) may be computationally expensive; it is recommended to acquire batches of samples(e.g. samples = 100).
    // The data acquired is organized in a matrix whose lines correspond to samples and the columns are as follows:
    //    * Sequence Number
    //    * 4 Digital Channels(always present)
    //    * 1-6 Analog Channels(as defined in the "start" method)
    // Example matrix for "analogChannels = [0, 1, 3]" used in "start" method:
    //    ==================  ========= ========= ========= ========= ======== ======== ========
    //    Sequence Number*    Digital 0 Digital 1 Digital 2 Digital 3 Analog 0 Analog 1 Analog 3              
    //    ==================  ========= ========= ========= ========= ======== ======== ========
    //    0                   
    //    1 
    //    (...)
    //    15
    //    0
    //    1
    //    (...)
    //    ==================  ========= ========= ========= ========= ======== ======== ========
    // Note:: * The sequence number overflows at 15 
    private void Read(int samples)
    {
        {
            int numberOfBytesToRead = 8;
            int numberOfActivechannels = 0;

            for (int i = 0; i < _analogChannels.Length; i++)
                if (_analogChannels[i])
                    numberOfActivechannels++;

            //if (numberOfActivechannels <= 4) 
            //    numberOfBytesToRead = (int)(Math.Round((12.0f+ 10.0f * numberOfActivechannels), MidpointRounding.AwayFromZero) / 8.0f);
            //else
            //    numberOfBytesToRead = (int)(Math.Round((52.0f + 6.0f * (numberOfActivechannels - 4)) / 8.0f));

            //dataAcquired = numpy.zeros((nSamples, 5 + nChannels))

            for (int i = 0; i < samples; i++)//for sample in range(nSamples):
            {
                int bufferSizeToRead = _serialPortStream.BytesToRead;
                if (bufferSizeToRead > 8)
                {
                    byte[] buffer = new byte[numberOfBytesToRead];
                    {
                        //byte[] test = new byte[bufferSizeToRead];
                        //_serialPortStream.Read(test, 0, bufferSizeToRead - 1);

                        
                        //int counter = 0;
                        //for (int p = bufferSizeToRead - 8; p < bufferSizeToRead; p++)
                        //{
                        //    buffer[counter] = test[p];
                        //    counter++;
                        //}
                        //Array.Resize(ref test, bufferSizeToRead - 8);
                        //int len = test.Length;

                        //while (!checkCRC4(buffer, numberOfBytesToRead) && len > 8)
                        //{
                        //    for (int l = 0; l < numberOfBytesToRead - 1; l++)
                        //        buffer[l] = buffer[l + 1];

                        //    len = test.Length;
                        //    buffer[7] = test[len - 1];
                        //    Array.Resize(ref test, len - 1);
                        //    len = test.Length;
                        //}
                    }

                    byte seq = 0;

                    {
                        //byte[] buffer = new byte[numberOfBytesToRead];

                        for (int n = 0; n < numberOfBytesToRead; n++)
                        {
                            byte[] tempBuffer = new byte[1];
                            if (_serialPortStream.Read(tempBuffer, 0, 1) != 1) return;
                            buffer[n] = tempBuffer[0];
                        }

                        seq = (byte)(buffer[numberOfBytesToRead - 1] >> 4);

                        //if (_serialPortStream.Read(buffer, 0, numberOfBytesToRead - 1) != numberOfBytesToRead - 1) 
                        //    return;

                        while (!checkCRC4(buffer, numberOfBytesToRead))
                        {
                            // if CRC check failed, try to resynchronize with the next valid frame
                            // checking with one new byte at a time
                            //memmove(buffer, buffer + 1, nBytes - 1);
                            //if (recv(buffer + nBytes - 1, 1) != 1) return int(it - frames.begin());   // a timeout has occurred

                            for (int l = 0; l < numberOfBytesToRead - 1; l++)
                                buffer[l] = buffer[l + 1];

                            byte[] tempBufferSizeOne = new byte[1];
                            if (_serialPortStream.Read(tempBufferSizeOne, 0, 1) != 1)
                                return;
                            buffer[numberOfBytesToRead - 1] = tempBufferSizeOne[0];
                        }
                    }

                    if (seq != (byte)(buffer[numberOfBytesToRead - 1] >> 4)) return;

                    byte crc = (byte)(buffer[numberOfBytesToRead - 1] & 0x0F);//    crc = decodedData[-1] & 0x0F
                    //byte seq = (byte)(buffer[numberOfBytesToRead - 1] >> 4);

                    Debug.Log("Seq = " + seq);
                    Debug.Log("crc = " + crc);

                    //buffer[numberOfBytesToRead - 1] = (byte)(buffer[numberOfBytesToRead - 1] & 0xF0);

                    //int x = 0;
                    //for (int m = 0; m < numberOfBytesToRead; m++)
                    //    for (int bit = 7; bit >= 0; bit--)
                    //    {
                    //        x = x << 1;
                    //        if ((x & 0x10) > 0)
                    //            x = x ^ 0x03;
                    //        x = x ^ ((buffer[m] >> bit) & 0x01);
                    //    }

                    //if (crc == (x & 0x0F))//if (checkCRC4(buffer, numberOfBytesToRead))
                    {
                        {
                            //        dataAcquired[sample, 0] = decodedData[-1] >> 4
                            //        dataAcquired[sample, 1] = decodedData[-2] >> 7 & 0x01
                            //        dataAcquired[sample, 2] = decodedData[-2] >> 6 & 0x01
                            //        dataAcquired[sample, 3] = decodedData[-2] >> 5 & 0x01
                            //        dataAcquired[sample, 4] = decodedData[-2] >> 4 & 0x01
                            //if (_analogChannels[0]) //        if nChannels > 0:
                            {
                                int j = ((buffer[6] & 0x0F) << 6) | (buffer[5] >> 2); //            dataAcquired[sample, 5] = ((decodedData[-2] & 0x0F) << 6) | (decodedData[-3] >> 2)
                                Debug.Log("analog[0] = " + j);
                                _analogChannelsResults[0] = j;
                            }

                            // ECG (Electrpcardiography)
                            if (numberOfActivechannels > 1) //if (_analogChannels[1]) //        if nChannels > 1:
                            {
                                int j = ((buffer[5] & 0x03) << 8) | (buffer[4]);    //            dataAcquired[sample, 6] = ((decodedData[-3] & 0x03) << 8) | decodedData[-4]
                                Debug.Log("analog[1] = " + j);
                                
                                // Transfer function [-1.47mV, +1.47mV] (micro Volt)
                                int VCC = 3; // Operating voltage
                                int ADC = j; // Value sampled form the channel
                                int n = 8; // Number of bits of the channel
                                int gECG = 1019; // sensor gain
                                float ECG_V = ((((float)ADC / (float)Math.Pow(2.0, (double)n)) - 0.5f) * (float) VCC) / (float)gECG;
                                Debug.Log("A2 (Volt) = " + ECG_V);
                                Debug.Log("A2 (milli Volt) = " + (ECG_V * 1000));
                                

                                _analogChannelsResults[1] = (int)(ECG_V * 1000);
                            }

                            // EDA (Electrodermal Activity) port A3
                            if (numberOfActivechannels > 2)//if (_analogChannels[2]) //        if nChannels > 2:
                            {
                                int j = ((buffer[3]) << 2) | (buffer[2] >> 6); //            dataAcquired[sample, 7] = (decodedData[-5] << 2) | (decodedData[-6] >> 6)
                                Debug.Log("analog[2] = " + j);

                                // Transfer function [0uS, 25uS] (micro Siemens)
                                int VCC = 3; // Operating voltage
                                int ADC = j; // Value sampled form the channel
                                int n = 8; // Number of bits of the channel
                                float EDA_uS = (((float)ADC / (float)Math.Pow(2.0, (double)n)) * (float) VCC) / 0.12f;
                                Debug.Log("A3 (micro Siemens) = " + EDA_uS);
                                Debug.Log("A3 (Siemens) = " + (EDA_uS * Math.Pow(10.0, -6)));
                                _analogChannelsResults[2] = (int) EDA_uS;
                            }

                            if (numberOfActivechannels > 3)//if (_analogChannels[3]) //        if nChannels > 3:
                            {
                                int j = ((buffer[2] & 0x3F) << 4) | (buffer[1] >> 4); //            dataAcquired[sample, 8] = ((decodedData[-6] & 0x3F) << 4) | (decodedData[-7] >> 4)
                                Debug.Log("analog[3] = " + j);
                                _analogChannelsResults[3] = j;
                            }

                            if (numberOfActivechannels > 4)//if (_analogChannels[4]) //        if nChannels > 4:
                            {
                                int j = ((buffer[1] & 0x0F) << 2) | (buffer[0] >> 6); //            dataAcquired[sample, 9] = ((decodedData[-7] & 0x0F) << 2) | (decodedData[-8] >> 6)
                                Debug.Log("analog[4] = " + j);
                                _analogChannelsResults[4] = j;
                            }

                            if (numberOfActivechannels > 1)//if (_analogChannels[5]) //        if nChannels > 5:
                            {
                                int j = (buffer[0] & 0x3F); //            dataAcquired[sample, 10] = decodedData[-8] & 0x3F 
                                Debug.Log("analog[5] = " + j);
                                _analogChannelsResults[5] = j;
                            }
                        }
                    }
                }
            }
        }

    }

    // CRC4 check function
    uint[] crc4tab = new uint[] { 0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2 };

    bool checkCRC4(byte[] data, int len)
    {
        byte[] copyData = new byte[len];
        Array.Copy(data, copyData, len);
        uint crc = 0;
        byte b;
        for (int i = 0; i < len - 1; i++)
        {
           b = copyData[i];
           crc = crc4tab[crc] ^ (UInt16)((UInt16)b >> (UInt16)4);
           crc = crc4tab[crc] ^ (UInt16)((UInt16)b & 0x0F);
        }

        // CRC for last byte
        b = copyData[len - 1];
        crc = crc4tab[crc] ^ (UInt16)((UInt16)b >> 4);
        crc = crc4tab[crc];

        return ((UInt16)crc == (UInt16)((UInt16)b & 0x0F));
    }

    private string Version()
    {
        // send(0x07);    // 0  0  0  0  0  1  1  1 - Send version string
        int versionAccessValue = 7;
        _serialPortStream.Write(BitConverter.GetBytes(versionAccessValue), 0, 4);

        System.Threading.Thread.Sleep(150);

        string receivedBufferEncoded = string.Empty;
        int bufferSizeToRead = _serialPortStream.BytesToRead;    
        byte[] buffer = new byte[bufferSizeToRead];
        if (bufferSizeToRead > 0)
        {
            _serialPortStream.Read(buffer, 0, bufferSizeToRead);
            receivedBufferEncoded = Encoding.ASCII.GetString(buffer, 0, bufferSizeToRead); 
            Debug.Log(receivedBufferEncoded);
        }

        return receivedBufferEncoded;
    }
}
