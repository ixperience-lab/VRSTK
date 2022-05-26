
#include "bitalino.h"

#include <stdio.h>

#ifdef _WIN32

#include <conio.h>

bool keypressed(void)
{
	return (_kbhit() != 0);
}

#else // Linux or Mac OS

#include <sys/select.h>

bool keypressed(void)
{
   fd_set   readfds;
   FD_ZERO(&readfds);
   FD_SET(0, &readfds);

   timeval  readtimeout;
   readtimeout.tv_sec = 0;
   readtimeout.tv_usec = 0;

   return (select(FD_SETSIZE, &readfds, NULL, NULL, &readtimeout) == 1);
}

#endif

int main()
{
   try
   {
      // uncomment this block to search for Bluetooth devices (Windows and Linux)
      /*
      BITalino::VDevInfo devs = BITalino::find();   
      for(int i = 0; i < devs.size(); i++)
      	printf("%s - %s\n", devs[i].macAddr.c_str(), devs[i].name.c_str());
      return 0;
      */

      puts("Connecting to device...");
      
      // use one of the lines below
      BITalino dev("20:19:07:00:81:7E");  // device MAC address (Windows and Linux)
      
      //BITalino dev("COM5");  // Bluetooth virtual COM port or USB-UART COM port (Windows)
      
      //BITalino dev("/dev/ttyUSB0");  // USB-UART device (Linux)
      //BITalino dev("/dev/rfcomm0");  // Bluetooth virtual serial port (Linux) 
      
      //BITalino dev("/dev/tty.usbserial-A1000QIz");  // USB-UART device (Mac OS)
      //BITalino dev("/dev/tty.bitalino-DevB");  // Bluetooth virtual serial port (Mac OS) 
      
      puts("Connected to device. Press Enter to exit.");

      std::string ver = dev.version();    // get device version string
      printf("BITalino version: %s\n", ver.c_str());

      dev.battery(10);  // set battery threshold (optional)

      dev.start(1000, {0, 1, 2, 3, 4, 5});   // start acquisition of all channels at 1000 Hz
      // use block below if your compiler doesn't support vector initializer lists
      /*
      BITalino::Vint chans;
      chans.push_back(0);
      chans.push_back(1);
      chans.push_back(2);
      chans.push_back(3);
      chans.push_back(4);
      chans.push_back(5);
      dev.start(1000, chans);
      */

      //dev.trigger({false, false, true, false});  // for original BITalino
      //dev.trigger({true, false});                // for BITalino 2
      // use block below if your compiler doesn't support vector initializer lists
      /*
      BITalino::Vbool outputs;
      outputs.push_back(false);
      outputs.push_back(false);
      outputs.push_back(true);
      outputs.push_back(false);
      dev.trigger(outputs);
      */

      BITalino::VFrame frames(100); // initialize the frames vector with 100 frames 
      int counter = 0; 
      do
      {
         dev.read(frames); // get 100 frames from device
         const BITalino::Frame &f = frames[0];  // get a reference to the first frame of each 100 frames block
         printf("%d - %d : %d %d %d %d ; %d %d %d %d %d %d\n",   // dump the first frame
                counter,
                f.seq,
                f.digital[0], f.digital[1], f.digital[2], f.digital[3],
                f.analog[0], f.analog[1], f.analog[2], f.analog[3], f.analog[4], f.analog[5]);
         counter++;
      } while (!keypressed()); // until a key is pressed

      dev.stop(); // stop acquisition
   } // dev is destroyed here (it goes out of scope)
   catch(BITalino::Exception &e)
   {
      printf("BITalino exception: %s\n", e.getDescription());
   }
   
   return 0;
}
