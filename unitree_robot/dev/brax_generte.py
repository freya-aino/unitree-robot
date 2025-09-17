from brax.io import mjcf, html
from pathlib import Path

mjcf_file_path = str(Path("/home/fey/Downloads/test-go2-mjcf-file.xml").resolve())

print(f"loading mjcf file from: {mjcf}")
scene = mjcf.load(mjcf_file_path)

print(scene)
