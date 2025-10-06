from cyclonedds import builtin, sub
import cyclonedds.domain as dds
import cyclonedds.core as core
import cyclonedds.idl as idl
from cyclonedds.sub import DataReader
from unitree_robot.common.datastructure import NETWORK_INTERFACE

if __name__ == "__main__":

    import os
    # os.environ["CYCLONEDDS_NETWORK_INTERFACE"] = NETWORK_INTERFACE.LAPTOP_1.value
    os.environ["CYCLONEDDS_URI"] = "./unitree_robot/common/cyclonedds.xml"
    # TODO, CHANGE THIS TO THE CORRECT GLOBAL RELATIVE PATH WHEN TRANSFERING IT TO A FUNCTION

    dp = dds.DomainParticipant()
    reader = builtin.BuiltinDataReader(
        subscriber_or_participant=dp,
        builtin_topic=builtin.BuiltinTopicDcpsTopic
    )

    while True:
        for sample in reader.take():
            print(f"topic: {sample.topic_name}")


    # cdds_monitor = CycloneDDSMonitor(
    #     network_interface=NETWORK_INTERFACE.LAPTOP_1.value
    # )

    # try:
    #     cdds_monitor.start()

    #     for _ in range(100):

    #         dat = cdds_monitor.get_current_data()

    #         print(dat)

    # except Exception as e:
    #     print(e.with_traceback)
    #     cdds_monitor.stop()
