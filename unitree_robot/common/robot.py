from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.core import channel

# NETWORK_INTERFACE = "lo"
NETWORK_INTERFACE = "enp0s13f0u1"

# initialize
channel.ChannelFactoryInitialize(0, NETWORK_INTERFACE)

# set channel
utlidar_publisher = channel.ChannelPublisher(
    name="rt/utlidar/switch",
    type=String_
)
utlidar_publisher.Init()
# channel.ChannelSubscriber()


low_cmd = std_msgs_msg_dds__String_()
low_cmd.data = "OFF"
utlidar_publisher.Write(low_cmd)
