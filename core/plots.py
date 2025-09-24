import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.phonon.plotter import PhononDosPlotter, PhononBSPlotter

def plot_phonon_dos_png(ph_dos) -> bytes:
    p = PhononDosPlotter()
    p.add_dos("Phonon DOS", ph_dos)
    ax = p.get_plot()
    ax.set_xlabel("Frequency (THz)"); ax.set_ylabel("DOS")
    fig = ax.get_figure()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def plot_phonon_band_png(ph_bs) -> bytes:
    p = PhononBSPlotter(ph_bs)
    ax = p.get_plot()
    ax.set_ylabel("Frequency (THz)")
    fig = ax.get_figure()
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()