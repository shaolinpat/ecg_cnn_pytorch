# tests/test_fetch_ptbxl.py

"""
Smoke test for scripts/fetch_ptbxl.py.

Notes
-----
    - Marked skipped by default: it would attempt network access.
    - Verifies the script starts and creates the expected output directory.
"""

import pytest

# import os
# import shutil
import subprocess
import sys

from pathlib import Path


@pytest.mark.skip(reason="Smoke test that triggers network call; run manually")
def test_fetch_ptbxl_smoke(tmp_path, monkeypatch):
    """
    Smoke test for fetch_ptbxl.py — does not check download success,
    only that the script executes and creates the expected directory.
    """
    script_path = Path("scripts/fetch_ptbxl.py")
    assert script_path.exists(), f"Script not found at {script_path}"

    # Redirect output location via env var only (no file mutation)
    out_dir = tmp_path / "ptbxl_test"
    monkeypatch.setenv("PTBXL_OUT_DIR", str(out_dir))

    # Run the script (allow failures — we just care if it tries)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Check it created (or at least initialized) the output directory
    assert out_dir.exists(), "Output directory was not created"

    # Accept logs on either stream and either path (S3 or wget fallback)
    combined = (result.stdout or "") + (result.stderr or "")
    assert ("Attempting AWS S3 sync" in combined) or (
        "falling back to wget" in combined
    )
