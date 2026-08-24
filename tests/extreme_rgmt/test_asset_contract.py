import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from contracts import G1_URDF_SHA256


def test_packaged_urdf_hash_and_mesh_closure():
    asset_dir = Path(__file__).resolve().parents[2] / "source/unitracker_lab/unitracker_lab/assets/g1"
    urdf = asset_dir / "g1_29dof_rev_1_0.urdf"
    manifest = json.loads((asset_dir / "asset_manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == G1_URDF_SHA256 == manifest["sha256"]
    tree = ET.parse(urdf)
    mesh_paths = {mesh.attrib["filename"] for mesh in tree.findall(".//mesh")}
    assert len(mesh_paths) == manifest["mesh_count"] == 35
    assert all((asset_dir / relative_path).is_file() for relative_path in mesh_paths)
