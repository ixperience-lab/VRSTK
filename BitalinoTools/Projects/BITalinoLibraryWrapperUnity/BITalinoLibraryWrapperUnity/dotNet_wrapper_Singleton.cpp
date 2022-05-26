/**
 * \file
 * \copyright  Copyright 2014 PLUX - Wireless Biosignals, S.A.
 * \author     Filipe Silva
 * \version    1.1
 * \date       July 2014
 * 
 * \section LICENSE
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 
 You should have received a copy of the GNU Lesser General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 */


#include "bitalino.h"

using namespace System;
using namespace System::Runtime::InteropServices;


/// <summary>The BITalino device class.</summary>
public ref class Bitalino
{
public:
// Type definitions

   ref struct DevInfo
   {
      /// <summary>MAC address of a Bluetooth device</summary>
      const String^ macAddr;
      /// <summary>name of a Bluetooth device</summary>
      const String^ name;

      DevInfo(const BITalino::DevInfo &dev) :
         macAddr(gcnew String(dev.macAddr.c_str())),
         name(gcnew String(dev.name.c_str())) {}
   };

   ref struct Frame
   {
      /// <summary>Frame sequence number (0..15)</summary>
      char  seq;
      /// <summary>Array of digital inputs states (false or true)</summary>
      array<bool>^  digital;
      /// <summary>Array of analog inputs values (0...1023 or 0...63)</summary>
      array<short>^ analog;

      Frame(void)
      {
         digital = gcnew array<bool>(4);
         analog  = gcnew array<short>(6);
      }
   };

   ref struct Exception : public ApplicationException
   {
   public:
      enum class Code
      {
         /// <summary>The object instance is closed</summary>
         INSTANCE_CLOSED = 0,

         /// <summary>The specified address is invalid</summary>
         INVALID_ADDRESS = 1,
         
         /// <summary>No Bluetooth adapter was found</summary>
         BT_ADAPTER_NOT_FOUND,
         
         /// <summary>The device could not be found</summary>
         DEVICE_NOT_FOUND,
         
         /// <summary>The computer lost communication with the device</summary>
         CONTACTING_DEVICE,
         
         /// <summary>The communication port does not exist or it is already being used</summary>
         PORT_COULD_NOT_BE_OPENED,
         
         /// <summary>The communication port could not be initialized</summary>
         PORT_INITIALIZATION,
         
         /// <summary>The device is not idle</summary>
         DEVICE_NOT_IDLE,
         
         /// <summary>The device is not in acquisition mode</summary>
         DEVICE_NOT_IN_ACQUISITION,
         
         /// <summary>Invalid parameter</summary>
         INVALID_PARAMETER,
      } code;

      Exception(BITalino::Exception &e) : ApplicationException(gcnew String(e.getDescription())), code((Code) e.code) {}
      Exception(String^ msg) : ApplicationException(msg), code(Code::INSTANCE_CLOSED) {}
	};

// Static methods

   /// <summary>Searches for Bluetooth devices in range.</summary>
   /// <returns>a list of found devices.</returns>
   /// <exception cref="Exception"><see cref="Exception::Code::PORT_INITIALIZATION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::BT_ADAPTER_NOT_FOUND"/></exception>
   static array<DevInfo^>^ find(void);
   static String^ _address;
   static void createBITalinoInstance(String^ address);


   static property Bitalino^ Instance { Bitalino^ get()
   {
       return % _instance;
   }}

   //// Instance methods

   ///// <summary>Connects to a BITalino device.</summary>
   ///// <param name="address">The device Bluetooth MAC address ("xx:xx:xx:xx:xx:xx") or a serial port ("COMx")</param>
   ///// <exception cref="Exception"><see cref="Exception::Code::PORT_COULD_NOT_BE_OPENED"/></exception>
   ///// <exception cref="Exception"><see cref="Exception::Code::PORT_INITIALIZATION"/></exception>
   ///// <exception cref="Exception"><see cref="Exception::Code::INVALID_ADDRESS"/></exception>
   ///// <exception cref="Exception"><see cref="Exception::Code::BT_ADAPTER_NOT_FOUND"/></exception>
   ///// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_FOUND"/></exception>
   //Bitalino(String^ address);

   ///// <summary>Disconnects from a BITalino device. If an aquisition is running, it is stopped.</summary> 
   //~Bitalino() { this->!Bitalino(); }   // destructor - called from Dispose()

   ///// <summary>Disconnects from a BITalino device. If an aquisition is running, it is stopped.</summary> 
   //!Bitalino()   // finalizer
   //{
   //   if (dev)
   //   {
   //      delete dev;
   //      dev = NULL;
   //   }
   //}


   /// <summary>Returns the device firmware version string.</summary> 
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   String^ version(void);

   /// <summary>Starts a live signal acquisition from all analog channels on the device at 1000 Hz.</summary> 
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void start(void);

   /// <summary>Starts a live signal acquisition from all analog channels on the device.</summary> 
   /// <param name="samplingRate">Sampling rate in Hz. Accepted values are 1, 10, 100 or 1000 Hz.</param>
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_PARAMETER"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void start(int samplingRate);

   /// <summary>Starts a live signal acquisition from the device.</summary> 
   /// <param name="samplingRate">Sampling rate in Hz. Accepted values are 1, 10, 100 or 1000 Hz.</param>
   /// <param name="channels">Set of channels to acquire. Accepted channels are 0, 1, 2, 3, 4, and 5.
   /// If this set is empty, all 6 analog channels will be acquired.</param>
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_PARAMETER"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void start(int samplingRate, array<int>^ channels);

   /// <summary>Starts a signal acquisition from the device.</summary> 
   /// <param name="samplingRate">Sampling rate in Hz. Accepted values are 1, 10, 100 or 1000 Hz.</param>
   /// <param name="channels">Set of channels to acquire. Accepted channels are 0, 1, 2, 3, 4, and 5.
   /// If this set is empty, all 6 analog channels will be acquired.</param>
   /// <param name="simulated">If true, start in simulated mode. Otherwise start in live mode.</param>
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_PARAMETER"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void start(int samplingRate, array<int>^ channels, bool simulated);

   /// <summary>Stops a signal acquisition.</summary>
   /// <remarks>This method must be called only during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IN_ACQUISITION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void stop(void);

   /// <summary>Reads acquisition frames from the device.
   /// This method does not return until all requested frames are received from the device.</summary>
   /// <param name="frames">Array of frames to be filled. This array cannot be empty.</param>
   /// <remarks>This method must be called only during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IN_ACQUISITION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void read(array<Frame^>^ frames);

   /// <summary>Sets the battery voltage threshold for the low-battery LED.</summary>
   /// <param name="value">Battery voltage threshold (0...63). 0 corresponds to 3.4 V and 63 corresponds to 3.8 V.</param>
   /// <remarks>This method cannot be called during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IDLE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_PARAMETER"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void battery(int value);

   /// <summary>Resets all digital outputs to low state.</summary>
   /// <remarks>This method must be called only during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IN_ACQUISITION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void trigger(void);

   /// <summary>Assigns the digital outputs states.</summary>
   /// <param name="digitalOutput">Array of booleans to assign to digital outputs, starting at output 0.
   /// This array cannot contain more than 4 elements.
   /// If the array contains less than 4 elements, the remaining outputs are set to low state.</param>
   /// <remarks>This method must be called only during an acquisition.</remarks>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_IN_ACQUISITION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_PARAMETER"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::CONTACTING_DEVICE"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INSTANCE_CLOSED"/></exception>
   void trigger(array<bool>^ digitalOutput);


private:
   // hewl
   static Bitalino _instance;

   // Instance methods
   Bitalino() {};
   /// <summary>Connects to a BITalino device.</summary>
   /// <param name="address">The device Bluetooth MAC address ("xx:xx:xx:xx:xx:xx") or a serial port ("COMx")</param>
   /// <exception cref="Exception"><see cref="Exception::Code::PORT_COULD_NOT_BE_OPENED"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::PORT_INITIALIZATION"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::INVALID_ADDRESS"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::BT_ADAPTER_NOT_FOUND"/></exception>
   /// <exception cref="Exception"><see cref="Exception::Code::DEVICE_NOT_FOUND"/></exception>
   Bitalino(String^ address);

   /// <summary>Disconnects from a BITalino device. If an aquisition is running, it is stopped.</summary> 
   //~Bitalino() { this->!Bitalino(); }   // destructor - called from Dispose()

   /// <summary>Disconnects from a BITalino device. If an aquisition is running, it is stopped.</summary> 
   //!Bitalino()   // finalizer
   //{
   //    if (dev)
   //    {
   //        delete dev;
   //        dev = NULL;
   //    }
   //}

   BITalino *dev;
   void validateDev(void)
   {
      if (!dev)  throw gcnew Exception("The object instance is closed.");
   }
};

array<Bitalino::DevInfo^>^ Bitalino::find(void)
{
   try
   {
      BITalino::VDevInfo devs = BITalino::find();
      array<Bitalino::DevInfo^>^ array_devs = gcnew array<Bitalino::DevInfo^>(devs.size());

      int i = 0;
      for(BITalino::VDevInfo::const_iterator it = devs.begin(); it != devs.end(); it++)
         array_devs[i++] = gcnew Bitalino::DevInfo(*it);

      return array_devs;
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::createBITalinoInstance(String^ address)
{
    IntPtr ip = Marshal::StringToHGlobalAnsi(address);
    const char* str = static_cast<const char*>(ip.ToPointer());

    try
    {
        dev = new BITalino(str);
    }
    catch (BITalino::Exception& e)
    {
        throw gcnew Exception(e);
    }
    finally
    {
        Marshal::FreeHGlobal(ip);
    }
}

Bitalino::Bitalino(String^ address)
{
   IntPtr ip = Marshal::StringToHGlobalAnsi(address);
   const char* str = static_cast<const char*>(ip.ToPointer());

   try
   {
      dev = new BITalino(str);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
   finally
   {
      Marshal::FreeHGlobal(ip);
   }
}

String^ Bitalino::version(void)
{
   validateDev();

   try
   {
      return gcnew String(dev->version().c_str());
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::start(void)
{
   validateDev();

   try
   {
      dev->start();
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::start(int samplingRate)
{
   validateDev();

   try
   {
      dev->start(samplingRate);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::start(int samplingRate, array<int>^ channels)
{
   validateDev();

   BITalino::Vint chans(channels->Length);
   for(int i = 0; i < channels->Length; i++)
      chans[i] = channels[i];
   try
   {
      dev->start(samplingRate, chans);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::start(int samplingRate, array<int>^ channels, bool simulated)
{
   validateDev();

   BITalino::Vint chans(channels->Length);
   for(int i = 0; i < channels->Length; i++)
      chans[i] = channels[i];
   try
   {
      dev->start(samplingRate, chans, simulated);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::stop(void)
{
   validateDev();

   try
   {
      dev->stop();
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::read(array<Frame^>^ frames)
{
   validateDev();

   if (frames->Length == 0)
      throw gcnew Exception(BITalino::Exception(BITalino::Exception::INVALID_PARAMETER));

   BITalino::VFrame xframes(frames->Length);
   try
   {
      dev->read(xframes);
      for(int i = 0; i < frames->Length; i++)
      {
         frames[i]->seq = xframes[i].seq;

         pin_ptr<bool> ppDig = &frames[i]->digital[0];
         memcpy(ppDig, xframes[i].digital, sizeof xframes[i].digital);

         pin_ptr<short> ppAn = &frames[i]->analog[0];
         memcpy(ppAn, xframes[i].analog, sizeof xframes[i].analog);
      }
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::battery(int value)
{
   validateDev();

   try
   {
      dev->battery(value);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::trigger(void)
{
   validateDev();

   try
   {
      dev->trigger();
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}

void Bitalino::trigger(array<bool>^ digitalOutput)
{
   validateDev();

   BITalino::Vbool dout(digitalOutput->Length);
   for(int i = 0; i < digitalOutput->Length; i++)
      dout[i] = digitalOutput[i];
   try
   {
      dev->trigger(dout);
   }
   catch (BITalino::Exception &e)
   {
      throw gcnew Exception(e);
   }
}
