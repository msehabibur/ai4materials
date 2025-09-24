# core/ct_bridge.py
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import streamlit as st
from pymatgen.core import Structure

DEFAULT_PORT = 8051

def _script_dir() -> Path:
    # Resolve project root (folder containing app.py and ct_app.py)
    return Path(__file__).resolve().parents[1]

def _ct_app_path() -> Path:
    return _script_dir() / "ct_app.py"

def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def _shared_paths():
    # Write to the Streamlit session temp dir by default
    if "workdir" in st.session_state:
        base = Path(st.session_state.workdir)
    else:
        base = Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    mson_path = base / "ct_shared_structure.json"
    return mson_path

def write_structure_to_mson(struct: Structure, mson_path: Path):
    mson_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mson_path, "w") as f:
        json.dump(struct.as_dict(), f)

def launch_ct_if_needed(mson_path: Path, port: int = DEFAULT_PORT):
    key = f"ct_server_{port}"
    # If already running and port is open, do nothing
    if _is_port_open("127.0.0.1", port):
        return

    ct_app = _ct_app_path()
    if not ct_app.exists():
        st.error(f"Crystal Toolkit server app not found: {ct_app}")
        return

    # Start ct_app.py as a child process with absolute paths
    cmd = [sys.executable, str(ct_app), "--mson", str(mson_path), "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_script_dir()),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    st.session_state[key] = proc

    # Wait up to ~5 seconds for the server to bind
    for _ in range(25):
        if _is_port_open("127.0.0.1", port):
            break
        time.sleep(0.2)

def embed_ct_iframe(port: int = DEFAULT_PORT, height: int = 560):
    if not _is_port_open("127.0.0.1", port):
        st.warning("Crystal Toolkit viewer is not reachable yet. Check that crystal_toolkit is installed.")
        return
    url = f"http://127.0.0.1:{port}"
    st.components.v1.iframe(src=url, height=height)

def stop_ct_server(port: int = DEFAULT_PORT):
    key = f"ct_server_{port}"
    proc = st.session_state.get(key)
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            time.sleep(0.3)
        except Exception:
            pass
        st.session_state[key] = None