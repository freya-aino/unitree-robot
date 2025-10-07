from pathlib import Path

from unitree_robot.common.cdds import get_all_cdds_topics
from unitree_robot.common.datastructure import NETWORK_INTERFACE
from mujoco import MjModel

if __name__ == "__main__":

    file_path = Path("external/files/urdf/Go2/go2.urdf").resolve()

    model = MjModel.from_xml_path(str(file_path))



    # all_topics = get_all_cdds_topics()
    # print(all_topics)
