import brax
from brax.envs import Env
from brax.io import mjcf, html
from pathlib import Path

from brax_envs import BACKENDS, CustomEnv

mjcf_file_path = Path("external/files/Go2/go2.xml").resolve()

# print(f"loading mjcf file from: {mjcf_file_path}")
# system = mjcf.load(str(mjcf_file_path))

# ---
