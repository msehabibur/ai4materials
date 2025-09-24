# ct_viewer.py — minimal, robust Crystal Toolkit viewer for CIF/POSCAR
from __future__ import annotations
import base64, os, socket
from typing import List, Optional

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

import crystal_toolkit.components as ctc
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice


def find_free_port(start=8050, tries=50) -> int:
    port = int(os.getenv("PORT", start))
    for _ in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError("No free port found.")


def parse_uploaded_structure(contents: str, filename: Optional[str]) -> Structure:
    _, content_string = contents.split(",", 1)
    decoded = base64.b64decode(content_string).decode("utf-8", errors="ignore")
    name = (filename or "").lower()

    if name.endswith(".cif"):
        return Structure.from_str(decoded, fmt="cif")

    low = decoded.lower()
    if "direct" in low or "cart" in low:
        return Structure.from_str(decoded, fmt="poscar")

    try:
        return Structure.from_str(decoded, fmt="cif")
    except Exception:
        return Structure.from_str(decoded, fmt="poscar")


def structure_caption(s: Structure) -> str:
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return (f"a,b,c = {a:.3f}, {b:.3f}, {c:.3f} Å | "
            f"α,β,γ = {alpha:.2f}, {beta:.2f}, {gamma:.2f}° | "
            f"natoms={len(s)}")


app = dash.Dash(__name__, prevent_initial_callbacks=True, title="Crystal Toolkit — CIF/POSCAR Viewer")

default_structures: List[Structure] = [
    Structure(Lattice.hexagonal(5, 3), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    Structure(Lattice.cubic(5), ["K", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
]

structure_component = ctc.StructureMoleculeComponent(default_structures[0], id="viewer")

upload = dcc.Upload(
    id="upload",
    children=html.Div(["Drag & Drop or ", html.A("Select a CIF/POSCAR")], style={"textAlign": "center"}),
    style={
        "width": "100%", "height": "80px", "lineHeight": "80px",
        "borderWidth": "2px", "borderStyle": "dashed", "borderRadius": "8px",
        "textAlign": "center", "margin": "10px 0",
    },
    multiple=False,
)

swap = html.Button("Swap Structure", id="swap", n_clicks=0, style={"margin": "8px 0"})
store_structs = dcc.Store(id="store_structs", data=[default_structures[0]])
store_index = dcc.Store(id="store_index", data=0)
caption = html.Div(id="caption", style={"marginTop": "6px", "color": "#666"})

layout = html.Div(
    [
        html.H3("Crystal Toolkit — Upload & View CIF/POSCAR"),
        html.P("Upload a CIF/POSCAR. Click “Swap Structure” to toggle if two are loaded."),
        upload, swap, structure_component.layout(), caption, store_structs, store_index
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "16px"},
)
ctc.register_crystal_toolkit(app=app, layout=layout)


@app.callback(Output("store_structs", "data"), Input("upload", "contents"), State("upload", "filename"))
def on_upload(contents, filename):
    if not contents:
        raise dash.exceptions.PreventUpdate
    up = parse_uploaded_structure(contents, filename)
    return [default_structures[0], up]


@app.callback(Output("store_index", "data"), Input("swap", "n_clicks"), State("store_structs", "data"), prevent_initial_call=True)
def on_swap(n_clicks, structs):
    if not structs or len(structs) == 1:
        return 0
    return (n_clicks or 0) % 2


@app.callback(Output(structure_component.id(), "data"), Output("caption", "children"),
              Input("store_structs", "data"), Input("store_index", "data"))
def update_view(structs, idx):
    if not structs:
        s = default_structures[0]
        return s, structure_caption(s)

    def to_struct(obj):
        return obj if isinstance(obj, Structure) else Structure.from_dict(obj)

    idx = max(0, min(int(idx or 0), len(structs) - 1))
    s = to_struct(structs[idx])
    return s, structure_caption(s)


if __name__ == "__main__":
    PORT = find_free_port(8050)
    print(f"✅ Crystal Toolkit running at http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)