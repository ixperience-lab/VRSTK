using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpOSC;

namespace RecordBitalinoFramesWithOSC
{
    class Program
    {
        static void Main(string[] args)
        {
            UDPSender sender = null;
            Bitalino dev = null;
            string macAddress = "20:19:07:00:81:7E";
            int batteryThreshold = 10;
            int samplingRate = 1000;
            int numberOfFrames = 100;
            int portNumber = 5555;
            string ipAddress = "127.0.0.1";
            try
            {
                // load mac address
                Console.WriteLine("Load mac address ...");
                System.IO.StreamReader streamReader = System.IO.File.OpenText("./BitalinoMacAddress.txt");
                string line = streamReader.ReadLine();
                if (line.Contains("="))
                    macAddress = line.Split('=')[1];

                streamReader.Close();

                // load configuration
                Console.WriteLine("Load configuration ...");
                streamReader = System.IO.File.OpenText("./RecordBitalinoFramesWithOSCConfiguration.txt");
                line = streamReader.ReadLine();
                if (line.Contains("="))
                    batteryThreshold = Convert.ToInt32(line.Split('=')[1]);
                line = streamReader.ReadLine();
                if (line.Contains("="))
                    samplingRate = Convert.ToInt32(line.Split('=')[1]);
                line = streamReader.ReadLine();
                if (line.Contains("="))
                    numberOfFrames = Convert.ToInt32(line.Split('=')[1]);
                line = streamReader.ReadLine();
                if (line.Contains("="))
                    portNumber = Convert.ToInt32(line.Split('=')[1]);
                line = streamReader.ReadLine();
                if (line.Contains("="))
                    ipAddress = line.Split('=')[1];

                streamReader.Close();

                Console.WriteLine("Create OSC sender ...");
                sender = new UDPSender(ipAddress, portNumber);

                Console.WriteLine("Connecting to device...");

                int counter = 0;

                dev = new Bitalino(macAddress);  // device MAC address
                                                          //Bitalino dev = new Bitalino("COM7");  // Bluetooth virtual COM port or USB-UART COM port

                Console.WriteLine("Connected to device. Press Enter to exit.");

                string ver = dev.version();    // get device version string
                Console.WriteLine("BITalino version: {0}", ver);

                Console.WriteLine("Set battery threshold ...");
                dev.battery(batteryThreshold);  // set battery threshold (optional)

                Console.WriteLine("Start record frames ...");
                dev.start(samplingRate, new int[] { 0, 1, 2, 3, 4, 5 });   // start acquisition of all channels at 1000 Hz

                Bitalino.Frame[] frames = new Bitalino.Frame[numberOfFrames];
                for (int i = 0; i < frames.Length; i++)
                    frames[i] = new Bitalino.Frame();   // must initialize all elements in the array

                do
                {
                    dev.read(frames); // get 100 frames from device
                    Bitalino.Frame f = frames[0];  // get a reference to the first frame of each 100 frames block

                    Console.WriteLine("{0} - Seq[{1}] : O[{2} {3} {4} {5}] ; A[{6} {7} {8} {9} {10} {11}]",   // dump the first frame
                                      counter,
                                      f.seq,
                                      f.digital[0], f.digital[1], f.digital[2], f.digital[3],
                                      f.analog[0], f.analog[1], f.analog[2], f.analog[3], f.analog[4], f.analog[5]);

                    string rawMessage = string.Format("{0} - Seq[{1}] : O[{2} {3} {4} {5}] ; A[{6} {7} {8} {9} {10} {11}]",   // dump the first frame
                                      counter,
                                      f.seq,
                                      f.digital[0], f.digital[1], f.digital[2], f.digital[3],
                                      f.analog[0], f.analog[1], f.analog[2], f.analog[3], f.analog[4], f.analog[5]);

                    if (counter == int.MaxValue)
                        counter = 0;
                    else
                        counter++;

                    //Console.WriteLine("Send frames to channel name /Bitalino/Frame ...");
                    var oscMessage = new SharpOSC.OscMessage("/Bitalino/Frame", rawMessage);
                    sender.Send(oscMessage.GetBytes());

                } while (!Console.KeyAvailable); // until a key is pressed

                Console.WriteLine("Stop device ...");
                dev.stop(); // stop acquisition

                dev.Dispose(); // disconnect from device

                Console.WriteLine("Close sender ...");
                sender.Close();
            }
            catch (Bitalino.Exception e)
            {
                Console.WriteLine("BITalino exception: {0}", e.Message);
                
                if (dev != null)
                    dev.Dispose(); // disconnect from device
                
                if (sender != null)
                    sender.Close();
            }
        }
    }
}
