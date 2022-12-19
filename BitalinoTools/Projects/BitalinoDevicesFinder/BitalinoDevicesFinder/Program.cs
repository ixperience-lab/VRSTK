using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BitalinoDevicesFinder
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Create stream writer!");
            System.IO.StreamWriter streamWriter = System.IO.File.CreateText("./BitalinoMacAddress.txt");
            Console.WriteLine("Search for bitalino devices!");
            Bitalino.DevInfo[] devs = Bitalino.find();
            Console.WriteLine("Save devices informations!");
            bool saved = false;
            foreach (Bitalino.DevInfo d in devs)
            {
                streamWriter.Write("{1}={0}", d.macAddr, d.name);
                Console.WriteLine("{1}={0}", d.macAddr, d.name);
                saved = true;
            }
            if (saved)
                Console.WriteLine("Devices Information saved!");

            Console.WriteLine("Close stream writer!");
            streamWriter.Close();
            return;
        }
    }
}
