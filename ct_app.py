# ct_app.py
import argparse
import json
import os
import time

import dash
from dash import html, dcc
from dash.dependencies import Input, Output

import crystal_toolkit.components as ctc
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice

parser = argparse.ArgumentParser()
parser.add_argument("--mson", type=str, default="ct_shared_structure.json")
parser.add_argument("--port", type=int, default=8051)
args = parser.parse_args()

MSON_PATH = os.path.abspath(args.mson)
PORT = int(args.port)

def _load_structure():
    try:
        with open(MSON_PATH, "r") as f:
            data = json.load(f)
        return Structure.from_dict(data)
    except Exception:
        return Structure(Lattice.cubic(4.2), ["Na", "K"], [[0, 0, 0], [0.5, 0.5, 0.5]])

initial_struct = _load_structure()
structure_component = ctc.StructureMoleculeComponent(initial_struct, id="ct_structure")

interval = dcc.Interval(id="poll", interval=1000, n_intervals=0)
layout = html.Div(
    [html.H3("Crystal Toolkit Viewer"), structure_component.layout(), interval],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "10px"},
)

app = dash.Dash(__name__, prevent_initial_callbacks=True, title="Crystal Toolkit Viewer")
ctc.register_crystal_toolkit(app=app, layout=layout)

_last_mtime = 0.0

@app.callback(Output(structure_component.id(), "data"), Input("poll", "n_intervals"))
def refresh_structure(_):
    global _last_mtime
    try:
        mtime = os.path.getmtime(MSON_PATH)
    except Exception:
        return dash.no_update
    if mtime <= _last_mtime:
        return dash.no_update
    _last_mtime = mtime
    try:
        with open(MSON_PATH, "r") as f:
            data = json.load(f)
        return Structure.from_dict(data)
    except Exception:
        return dash.no_update

if __name__ == "__main__":
    time.sleep(0.1)
    app.run_server(debug=False, port=PORT, host="127.0.0.1")