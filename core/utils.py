import os, tempfile

def ensure_tmpdir(prefix: str = "tmp_") -> str:
    path = tempfile.mkdtemp(prefix=prefix)
    os.makedirs(path, exist_ok=True)
    return path