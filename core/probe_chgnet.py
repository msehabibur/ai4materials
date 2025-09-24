import io
import os
import tempfile


import matplotlib
matplotlib.use("Agg")




def ensure_tmpdir(prefix: str = "tmp_") -> str:
    path = tempfile.mkdtemp(prefix=prefix)
    os.makedirs(path, exist_ok=True)
    return path




def write_png(fig) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()