from __future__ import annotations
import json, uuid, io, warnings
import streamlit as st
from pymatgen.core import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write as ase_write

warnings.simplefilter("ignore")

def render_structure_viewer(pmg_obj, height: int = 480):
    """
    - Structure -> CIF (3Dmol 'cif')
    - Molecule  -> XYZ (3Dmol 'xyz')
    - Else try ASE -> XYZ
    """
    if pmg_obj is None:
        st.info("Upload a crystal in the sidebar to visualize.")
        return

    fmt = None
    text = None
    show_cell = False

    try:
        if isinstance(pmg_obj, Structure):
            text = pmg_obj.to(fmt="cif")
            fmt = "cif"
            show_cell = True
        elif isinstance(pmg_obj, Molecule):
            text = pmg_obj.to(fmt="xyz")
            fmt = "xyz"
        else:
            atoms = AseAtomsAdaptor.get_atoms(pmg_obj)
            buf = io.StringIO()
            ase_write(buf, atoms, format="xyz")
            text = buf.getvalue()
            fmt = "xyz"
    except Exception as exc:
        st.error(f"Viewer failed to serialize structure: {exc}")
        return

    container_id = f"viewer-{uuid.uuid4().hex}"
    data_json = json.dumps(text)

    html = f"""
    <div id="{container_id}" style="height: {height}px; position: relative;"></div>
    <script src="https://3dmol.csb.pitt.edu/build/3Dmol.js"></script>
    <script>
      (function() {{
        var viewer = $3Dmol.createViewer('{container_id}', {{backgroundColor: 'white'}});
        var data = {data_json};
        viewer.addModel(data, '{fmt}');
        viewer.setStyle({{}}, {{stick: {{radius: 0.18}}, sphere: {{scale: 0.25}}}});
        {"viewer.addUnitCell();" if show_cell else ""}
        viewer.zoomTo();
        viewer.render();
      }})();
    </script>
    """
    st.components.v1.html(html, height=height)
