import pytest
import os
import shutil
import subprocess


@pytest.mark.skip(reason="Smoke test that triggers network call; run manually")
def test_fetch_ptbxl_smoke(tmp_path):
    """
    Smoke test for fetch_ptbxl.py — does not check download success,
    only that the script executes and creates the expected directory.
    """
    # Set up temp output dir and override PTB-XL default location
    out_dir = tmp_path / "ptbxl_test"
    os.environ["PTBXL_OUT_DIR"] = str(out_dir)

    # Copy fetch script into tmp path so we can modify it
    script_path = tmp_path / "fetch_ptbxl.py"
    original_script = "scripts/fetch_ptbxl.py"
    shutil.copy(original_script, script_path)

    # Patch the script to use the new output path
    with open(script_path, "r") as f:
        code = f.read()
    code = code.replace('out = "data/ptbxl"', f'out = "{out_dir}"')
    with open(script_path, "w") as f:
        f.write(code)

    # Run the script (allow failures — we just care if it tries)
    result = subprocess.run(
        ["python", str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Check it created the output directory
    assert out_dir.exists(), "Output directory was not created"

    # Optionally check that it printed expected log lines
    output = result.stdout.decode()
    assert "Attempting AWS S3 sync" in output or "falling back to wget" in output
