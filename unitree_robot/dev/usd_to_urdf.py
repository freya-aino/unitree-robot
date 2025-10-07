from pathlib import Path
from nvidia.srl.from_usd.to_urdf import UsdToUrdf
from pxr import Usd

usd_source_file_path = Path("external/unitree-robot-models/Go2/usd/go2.usd").resolve()
target_urdf_file_path = Path("external/files/urdf/Go2/").resolve()

print(f"extracting usd from: {usd_source_file_path}")
usd_stage = Usd.Stage.Open(str(usd_source_file_path))

print(f"writing urdf to: {target_urdf_file_path}")

UsdToUrdf(stage=usd_stage).save_to_file(str(target_urdf_file_path))
