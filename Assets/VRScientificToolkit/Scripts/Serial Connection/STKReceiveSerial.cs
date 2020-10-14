using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO.Ports;
using Unity.Jobs;

namespace STK
{
    ///<summary>Reads a bytestream from a serial port.</summary>
    public class STKReceiveSerial : MonoBehaviour
    {

        public string port;
        public int baudRate;

        private SerialPort stream;
        public string currentValue;

        // Use this for initialization
        void Start()
        {
            stream = new SerialPort(port, baudRate);

            stream.Open();
        }

        // Update is called once per frame
        void Update()
        {
            if (stream.IsOpen)
            {
                if (stream.BytesToRead != 0)
                {
                    currentValue = stream.ReadLine();
                }
            }
        }
    }
}

