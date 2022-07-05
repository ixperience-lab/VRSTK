# Imports
from pylsl import StreamInlet, resolve_stream
from pythonosc import udp_client

class Lsl():

    def recv_data_unspecified_OS_stream(self, client):
        # Resolve an available OpenSignals stream
        print("# Looking for an available OpenSignals stream...")
        self.os_stream = resolve_stream("name", "OpenSignals")

        # Create an inlet to receive signal samples from the stream
        self.inlet = StreamInlet(self.os_stream[0])

        try:
            while True:
                # Receive samples
                sample, timestamp = self.inlet.pull_sample()
                print(timestamp, sample)
                #{0} - Seq[{1}] : O[{2} {3} {4} {5}] ; A[{6} {7} {8} {9} {10} {11}]"
                message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} 0.0 0.0 0.0 0.0 0.0 0.0]".format(timestamp, sample[0], sample[1])
                if len(sample) == 3:
                    message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} {3} 0.0 0.0 0.0 0.0]".format(timestamp, sample[0], sample[1], sample[2])
                if len(sample) == 4:
                    message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} {3} {4} 0.0 0.0 0.0]".format(timestamp, sample[0], sample[1], sample[2], sample[3])
                if len(sample) == 5:
                    message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} {3} {4} {5} 0.0 0.0]".format(timestamp, sample[0], sample[1], sample[2], sample[3], sample[4])
                if len(sample) == 6:
                    message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} {3} {4} {5} {6} 0.0]".format(timestamp, sample[0], sample[1], sample[2], sample[3], sample[4], sample[5])
                if len(sample) == 7:
                    message = "{0} - Seq[{1}] : O[false false false false] ; A[{2} {3} {4} {5} {6} {7}]".format(timestamp, sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6])
                print(message)
                client.send_message("/Bitalino/OpenSinglesStream", message)
        except KeyboardInterrupt:
            self.inlet.close_stream()


    def recv_data_PLUX_device(self, mac_address):
        # Resolve stream
        print("# Looking for an available OpenSignals stream from the specified device...")
        self.os_stream = resolve_stream("type", mac_address)

        # Create an inlet to receive signal samples from the stream
        self.inlet = StreamInlet(self.os_stream[0])

        try:
            while True:
                # Receive samples
                samples, timestamp = self.inlet.pull_sample()
                print(timestamp, samples)
        except KeyboardInterrupt:
            self.inlet.close_stream()


    def recv_data_host(self, hostname):
        # Resolve stream
        print("# Looking for an available OpenSignals stream from the specified host...")
        self.os_stream = resolve_stream("hostname", hostname)

        # Create an inlet to receive signal samples from the stream
        self.inlet = StreamInlet(self.os_stream[0])

        try:
            while True:
                # Receive samples
                samples, timestamp = self.inlet.pull_sample()
                print(timestamp, samples)
        except KeyboardInterrupt:
            self.inlet.close_stream()


    def recv_stream_metadata(self):
        # Get information about the stream
        self.stream_info = self.inlet.info()

        # Get individual attributes
        stream_name = self.stream_info.name()
        stream_mac = self.stream_info.type()
        stream_host = self.stream_info.hostname()
        stream_n_channels = self.stream_info.channel_count()

        # Store sensor channel info & units in the dictionary
        stream_channels = dict()
        channels = self.stream_info.desc().child("channels").child("channel")

        # Loop through all available channels
        for i in range(stream_n_channels - 1):

            # Get the channel number (e.g. 1)
            channel = i + 1

            # Get the channel type (e.g. ECG)
            sensor = channels.child_value("sensor")

            # Get the channel unit (e.g. mV)
            unit = channels.child_value("unit")

            # Store the information in the stream_channels dictionary
            stream_channels.update({channel: [sensor, unit]})
            channels = channels.next_sibling()



def show_menu():
    print('')
    for id in MENU_IMPUT.keys():
        print (str(id) + ' | ' + MENU_IMPUT[id])

def process_action(user_action, client):
    if user_action == '0':
        lsl.recv_data_unspecified_OS_stream(client)        
    elif user_action == '1':
        mac_address = str(input('MAC address: '))
        lsl.recv_data_PLUX_device(mac_address)
    elif user_action == '2':
        hostname = str(input('Host name: '))
        lsl.recv_data_host(hostname)
    elif user_action == '3':
        lsl.recv_stream_metadata()


if __name__ == "__main__":
    MENU_IMPUT = {0: 'Receving Data From an Unspecified OpenSignals Stream',
                  1: 'Receving Data From a Specific PLUX Device in an OpenSignals Stream',
                  2: 'Receving Data From a Specific Host Providing the OpenSignals Stream',
                  3: 'Receving Stream Metadata'
                  }

    lsl = Lsl()
    client = udp_client.SimpleUDPClient("127.0.0.1", 5555)

    while True:
        show_menu()
        user_action = str(input('New action: '))
        process_action(user_action, client)



