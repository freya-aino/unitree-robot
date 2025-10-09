from pathlib import Path

from unitree_robot.common.cdds import get_all_cdds_topics
from unitree_robot.common.datastructure import NETWORK_INTERFACE
# from mujoco import MjModel
from brax.io import mjcf

from mujoco import MjModel, MjData, Renderer

if __name__ == "__main__":

    file_path = Path("external/files/mjcf/Go2/go2.xml").resolve()

    # model = MjModel.from_xml_path(str(file_path))

    # mjcf.load(str(file_path))
    mj_model = MjModel.from_xml_path(str(file_path), assets)
    mj_data = MjData(mj_model)
    renderer = Renderer(mj_model)

    # all_topics = get_all_cdds_topics()
    # print(all_topics)
