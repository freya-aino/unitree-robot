from pathlib import Path

from lightwheel.srl.from_usd.to_mjcf import UsdToMjcf
from pxr import Usd

usd_source_file_path = Path("external/unitree-robot-models/Go2/usd/go2.usd").resolve()
target_mjcf_file_path = Path("external/files/Go2/").resolve()

print(f"extracting usd from: {usd_source_file_path}")
usd_stage = Usd.Stage.Open(str(usd_source_file_path))
mjcf_file = UsdToMjcf(stage=usd_stage).save_to_file(str(target_mjcf_file_path))

# from nvidia.srl.from_usd.to_urdf import UsdToUrdf

# UsdToUrdf.init_from_file(usd_file_path).save_to_file(urdf_file_path)

import lightwheel.srl.from_usd.to_mjcf
