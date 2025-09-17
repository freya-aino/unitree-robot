from pathlib import Path

usd_file_path = Path("unitree_robot/unitree_model/Go2/usd/go2.usd").resolve()
urdf_file_path = Path("unitree_robot/unitree_model_extracted/Go2/").resolve()

print(usd_file_path)

from nvidia.srl.from_usd.to_urdf import UsdToUrdf

UsdToUrdf.init_from_file(usd_file_path).save_to_file(urdf_file_path)
