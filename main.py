from cyclonedds import builtin, sub
import cyclonedds.domain as dds
import cyclonedds.core as core
import cyclonedds.idl as idl
from cyclonedds.sub import DataReader
from unitree_robot.common.cdds import get_all_cdds_topics
from unitree_robot.common.datastructure import NETWORK_INTERFACE

if __name__ == "__main__":

    all_topics = get_all_cdds_topics()

    print(all_topics)

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
