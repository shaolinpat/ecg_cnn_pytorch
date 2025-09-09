# tests/test_data_utils.py

"""
Tests for ecg_cnn.data.data_utils.

Covers
------
    - build_full_X_y(): path validation, minimal smoke, and error paths
    - PTBXLFullDataset: __init__/__len__/__getitem__ end-to-end
    - select_primary_label(): prioritization and input validation
    - aggregate_diagnostic(): input validation and restrict behavior
    - load_ptbxl_meta(): file presence, parsing, and happy path
    - load_ptbxl_sample(): path normalization, branch coverage, and errors
    - load_ptbxl_full(): parameter validation and data loading branches
    - raw_to_five_class(): string/dict parsing and mapping to 5-class labels
"""

import numpy as np
import os
import pandas as pd
import pytest

# Ensure optional deps cleanly skip if missing (no hard fail during import)
wfdb = pytest.importorskip("wfdb", reason="wfdb not installed")
torch = pytest.importorskip("torch", reason="torch not installed")

from pathlib import Path
from unittest.mock import patch
from ecg_cnn.data.dataset import PTBXLFullDataset
from ecg_cnn.data.data_utils import (
    build_full_X_y,
    select_primary_label,
    aggregate_diagnostic,
    load_ptbxl_meta,
    load_ptbxl_sample,
    load_ptbxl_full,
    raw_to_five_class,
    LABEL2IDX,
    FIVE_SUPERCLASSES,
)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def write_mock_ptb_record(dir_path, ecg_id):
    """Create fake WFDB record files for a given ecg_id under dir_path.

    Returns
    -------
    str
        Relative WFDB path without extension, e.g. '10000/10001_lr'.
    """
    record_dir = os.path.join(dir_path, f"{ecg_id // 1000 * 1000:05d}")
    os.makedirs(record_dir, exist_ok=True)

    record_name = f"{ecg_id}_lr"
    signal = np.random.normal(0, 0.001, size=(1000, 12))  # tiny noise

    # Save and restore working directory around call
    prev_dir = os.getcwd()
    os.chdir(record_dir)
    try:
        wfdb.wrsamp(
            record_name,
            fs=500,
            units=["mV"] * 12,
            sig_name=[f"lead{i}" for i in range(12)],
            p_signal=signal,
        )
    finally:
        os.chdir(prev_dir)

    return f"{int(ecg_id) // 1000 * 1000:05d}/{record_name}"


def create_dummy_scp_statements(path):
    """
    Creates a minimal dummy `scp_statements.csv` file at the given path.

    This file mimics the format expected by the PTB-XL metadata loader,
    containing a single diagnostic SCP code ("MI") mapped to its diagnostic class.
    Useful for unit testing functions that depend on `scp_statements.csv`.

    Parameters
    ----------
    path : Path or str
        Directory where the dummy `scp_statements.csv` will be created.
    """
    df = pd.DataFrame(
        {"scp_code": ["MI"], "diagnostic": [1], "diagnostic_class": ["MI"]}
    ).set_index("scp_code")
    df.to_csv(path / "scp_statements.csv")


def _write_csv(path: Path, rows: int, cols: int):
    """Write a numeric CSV with given shape (rows x cols), no header."""
    lines = [",".join("0" for _ in range(cols)) for _ in range(rows)]
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------
# def build_full_X_y(meta_csv, scp_csv, ptb_path):
# ------------------------------------------------------------------------------


def test_build_full_X_y_missing_meta_csv(tmp_path):
    """Should raise FileNotFoundError if meta CSV is missing."""
    meta = tmp_path / "meta.csv"
    scp = tmp_path / "scp.csv"
    scp.write_text("dummy\n")  # create dummy scp

    ptb_dir = tmp_path / "ptb"
    ptb_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"^Metadata CSV not found"):
        build_full_X_y(str(meta), str(scp), str(ptb_dir))


def test_build_full_X_y_missing_scp_csv(tmp_path):
    """Should raise FileNotFoundError if scp CSV is missing."""
    meta = tmp_path / "meta.csv"
    meta.write_text("dummy\n")

    scp = tmp_path / "scp.csv"  # intentionally not created

    ptb_dir = tmp_path / "ptb"
    ptb_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"^SCP statements CSV not found"):
        build_full_X_y(str(meta), str(scp), str(ptb_dir))


def test_build_full_X_y_missing_ptb_path(tmp_path):
    """Should raise NotADirectoryError if ptb dir doesn't exist."""
    # Create dummy  meta
    meta = tmp_path / "meta.csv"
    meta.write_text("dummy\n")
    # Create dummy scp CSV
    scp = tmp_path / "scp.csv"
    scp.write_text("dummy,desc\n")

    bad_dir = tmp_path / "nonexistent" / "dir"

    with pytest.raises(NotADirectoryError, match=r"^PTB-XL root directory not found"):
        build_full_X_y(meta, scp, bad_dir)


def test_build_full_X_y_smoke(tmp_path, patch_paths):
    # Create minimal meta CSV
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "ecg_id,filename_lr,scp_codes\n"
        "10001,records100/10000/10001_lr,\"['NORM']\"\n"
    )

    # Create dummy scp CSV
    scp = tmp_path / "scp.csv"
    scp.write_text("dummy,desc\n")

    # Use fixture-patched PTB-XL root; skip if sample record files are absent
    _, _, _, _, _, _, ptb_dir = patch_paths
    rel = Path("records100/10000/10001_lr")
    hea = ptb_dir / f"{rel}.hea"
    dat = ptb_dir / f"{rel}.dat"
    if not (hea.exists() and dat.exists()):
        pytest.skip("Required PTB-XL sample record files not found in PTBXL_DATA_DIR")

    X, y, meta_kept = build_full_X_y(meta, scp, ptb_dir)

    assert X.shape[1:] == (12, 1000)
    assert len(X) == len(y) == len(meta_kept)


@patch("ecg_cnn.data.data_utils.wfdb.rdsamp")
def test_valid_record_is_loaded(mock_rdsamp, tmp_path):
    # Arrange
    mock_rdsamp.return_value = (np.random.rand(1000, 12), {})  # Fake ECG signal
    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text("ecg_id,filename_lr,scp_codes\n1,test_path,\"['NORM']\"\n")

    scp_csv = tmp_path / "scp.csv"
    scp_csv.write_text("dummy\n")

    ptb_path = tmp_path
    (ptb_path / "test_path.hea").write_text(
        "dummy\n"
    )  # avoid file-not-found if not mocked deeply

    # Act
    X, y, meta = build_full_X_y(meta_csv, scp_csv, ptb_path)

    # Assert
    assert X.shape == (1, 12, 1000)
    assert y[0] == LABEL2IDX["NORM"]


def test_build_full_X_y_minimal_valid_record(tmp_path):
    """Test build_full_X_y() with one valid mocked record."""

    ecg_id = 12345
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    # 1. Create mock WFDB record and get its relative path
    rel_path = write_mock_ptb_record(ptb_dir, ecg_id)

    # 2. Create minimal metadata CSV with required fields
    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": [["NORM"]]}
    ).set_index("ecg_id")
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    # 3. Create dummy scp_statements CSV
    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    # 4. Call build_full_X_y and check results
    X, y, meta = build_full_X_y(meta_csv, scp_csv, ptb_dir)

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 12, 1000)

    assert isinstance(y, np.ndarray)
    assert y.shape == (1,)
    assert y[0] in range(5)

    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == 1


def test_build_full_X_y_skips_invalid_scp_codes(tmp_path):
    """Should skip rows with unrecognized scp_codes."""
    ecg_id = 99999
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    rel_path = write_mock_ptb_record(ptb_dir, ecg_id)

    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": [["FAKE_LABEL"]]}
    ).set_index("ecg_id")
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    scp_df = pd.DataFrame({"description": ["Unknown"]}, index=["FAKE_LABEL"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    with pytest.raises(ValueError, match=r"^No valid records"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_skips_unreadable_record(tmp_path):
    """Should skip unreadable WFDB records."""
    ecg_id = 77777
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    rel_path = write_mock_ptb_record(ptb_dir, ecg_id)

    # Delete the .dat file to simulate failure
    record_dir = ptb_dir / rel_path.split("/")[0]
    dat_path = record_dir / f"{ecg_id}_lr.dat"
    if dat_path.exists():
        dat_path.unlink()

    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": [["NORM"]]}
    ).set_index("ecg_id")
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    with pytest.raises(ValueError, match=r"^No valid records"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_empty_filename_lr(tmp_path):
    """Should skip rows with missing filename_lr (empty)."""
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text("ecg_id,filename_lr,scp_codes\n123,,\"['NORM']\"\n")

    scp_csv = tmp_path / "scp.csv"
    scp_csv.write_text("code,description\nNORM,Normal\n")

    with pytest.raises(ValueError, match=r"^No valid records were loaded"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_missing_filename_lr(tmp_path):
    """Should raise when 'filename_lr' is absent."""
    ecg_id = 12345
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    # No filename_lr included
    meta_df = pd.DataFrame({"ecg_id": [ecg_id], "scp_codes": [["NORM"]]}).set_index(
        "ecg_id"
    )
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    with pytest.raises(ValueError, match=r"^No valid records"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_stringified_scp_codes(tmp_path):
    """Covers the case where scp_codes are stored as strings, not dicts."""

    ecg_id = 45678
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    # Create mock WFDB record
    rel_path = write_mock_ptb_record(ptb_dir, ecg_id)

    # scp_codes is a string like "{'NORM': 1.0}" → tests json.loads path
    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": ["{'NORM': 1.0}"]}
    ).set_index("ecg_id")
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    X, y, meta = build_full_X_y(meta_csv, scp_csv, ptb_dir)

    assert isinstance(X, np.ndarray)
    assert X.shape == (1, 12, 1000)

    assert isinstance(y, np.ndarray)
    assert y.shape == (1,)
    assert y[0] in range(5)

    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == 1


# ------------------------------------------------------------------------------
# class PTBXLFullDataset(Dataset):
# ------------------------------------------------------------------------------


def test_PTBXLFullDataset_end_to_end(tmp_path):
    """Covers __init__, __len__, and __getitem__ for PTBXLFullDataset."""

    ecg_id = 12345
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    # Create mock WFDB record and get its relative path
    rel_path = write_mock_ptb_record(ptb_dir, ecg_id)

    # Create meta CSV
    meta_df = pd.DataFrame(
        {"ecg_id": [ecg_id], "filename_lr": [rel_path], "scp_codes": [["NORM"]]}
    ).set_index("ecg_id")
    meta_csv = tmp_path / "meta.csv"
    meta_df.to_csv(meta_csv)

    # Create dummy scp CSV
    scp_df = pd.DataFrame({"description": ["Normal"]}, index=["NORM"])
    scp_csv = tmp_path / "scp.csv"
    scp_df.to_csv(scp_csv)

    # Instantiate dataset
    ds = PTBXLFullDataset(meta_csv, scp_csv, ptb_dir)

    # Hit __len__ and __getitem__
    assert len(ds) == 1
    X, y = ds[0]
    assert isinstance(X, torch.Tensor)
    assert X.shape == (12, 1000)
    assert isinstance(y, int)


# ------------------------------------------------------------------------------
# def select_primary_label(label_list):
# ------------------------------------------------------------------------------


def test_select_primary_label_single_match():
    assert select_primary_label(["MI"]) == "MI"


def test_select_primary_label_multiple_matches_priority_order():
    # MI comes before STTC in FIVE_SUPERCLASSES
    assert select_primary_label(["STTC", "MI"]) == "MI"
    assert select_primary_label(["CD", "NORM", "HYP"]) == "CD"


def test_select_primary_label_only_unknown_labels():
    assert select_primary_label(["XYZ", "FOO"]) == "Unknown"


def test_select_primary_label_empty_list():
    assert select_primary_label([]) == "Unknown"


def test_select_primary_label_set_input():
    assert select_primary_label({"MI", "CD"}) == "CD"  # CD comes first in priority


def test_select_primary_label_tuple_input():
    assert select_primary_label(("NORM", "MI")) == "MI"


def test_select_primary_label_type_error_on_non_iterable():
    with pytest.raises(TypeError, match=r"^label_list must be a list, set, or tuple"):
        select_primary_label("MI")  # not a list-like structure


def test_select_primary_label_type_error_on_non_string_elements():
    with pytest.raises(TypeError, match=r"^All elements in label_list must be strings"):
        select_primary_label(["MI", 123])


# ------------------------------------------------------------------------------
# def aggregate_diagnostic(codes, agg_df, restrict=True):
# ------------------------------------------------------------------------------


def test_aggregate_diagnostic_codes_not_a_dictionary():
    dummy_df = pd.DataFrame(columns=["scp_code", "diagnostic", "diagnostic_class"])
    with pytest.raises(ValueError, match=r"^codes must be a dictionary"):
        aggregate_diagnostic('{"NORM": 1.0, "MI": 0.0}', "not_a_df", restrict=True)


def test_aggregate_diagnostic_not_dataframe():
    with pytest.raises(ValueError, match=r"^agg_df must be a pandas DataFrame"):
        aggregate_diagnostic({"NORM": 1.0, "MI": 0.0}, "not_a_df")


def test_aggregate_diagnostic_unknown_code():
    dummy_df = pd.DataFrame(
        {
            "scp_code": ["FAKE"],
            "diagnostic": [1],  # Diagnostic, but class is not in target list
            "diagnostic_class": ["ALIEN"],
        }
    ).set_index("scp_code")
    result = aggregate_diagnostic({"FAKE": 1.0}, dummy_df, restrict=True)
    assert result == ["Unknown"]


def test_aggregate_diagnostic_restrict_false_unusual_code():
    dummy_df = pd.DataFrame(
        {
            "scp_code": ["FAKE"],
            "diagnostic": [1],  # Diagnostic, but class is not in target list
            "diagnostic_class": ["ALIEN"],
        }
    ).set_index("scp_code")
    result = aggregate_diagnostic({"FAKE": 1.0}, dummy_df, restrict=False)
    assert result == ["ALIEN"]


def test_aggregate_diagnostic_series_true_and_false_paths():
    # Unique index -> .loc['SCP_A'] returns a Series
    data = [
        ("SCP_A", 1, FIVE_SUPERCLASSES[0]),  # diagnostic==1 -> TRUE branch
        ("SCP_B", 0, FIVE_SUPERCLASSES[1]),  # diagnostic==0 -> FALSE branch
    ]
    df = pd.DataFrame(
        data, columns=["code", "diagnostic", "diagnostic_class"]
    ).set_index("code")

    # TRUE branch: diagnostic==1
    out_true = aggregate_diagnostic({"SCP_A": 1.0}, df, restrict=True)
    assert out_true == [FIVE_SUPERCLASSES[0]]

    # FALSE branch: diagnostic==0 -> ignored -> returns ["Unknown"]
    out_false = aggregate_diagnostic({"SCP_B": 1.0}, df, restrict=True)
    assert out_false == ["Unknown"]


def test_aggregate_diagnostic_dataframe_from_duplicate_index_false_path_and_restrict_off():
    # Duplicate index -> .loc['SCP_DUP'] returns a DataFrame (not a Series) -> FALSE branch
    dup = [
        ("SCP_DUP", 1, FIVE_SUPERCLASSES[2]),
        ("SCP_DUP", 1, FIVE_SUPERCLASSES[3]),
    ]
    dup_df = pd.DataFrame(
        dup, columns=["code", "diagnostic", "diagnostic_class"]
    ).set_index("code")

    out_dup = aggregate_diagnostic({"SCP_DUP": 1.0}, dup_df, restrict=True)
    # Not a Series -> branch is false -> nothing appended -> ["Unknown"]
    assert out_dup == ["Unknown"]

    # Also verify restrict=False can include non-superclass labels
    df2 = pd.DataFrame(
        [("SCP_X", 1, "OTHER")], columns=["code", "diagnostic", "diagnostic_class"]
    ).set_index("code")
    out_all = aggregate_diagnostic({"SCP_X": 1.0}, df2, restrict=False)
    assert out_all == ["OTHER"]


# ------------------------------------------------------------------------------
# def load_ptbxl_meta(ptb_path):
# ------------------------------------------------------------------------------


def test_load_ptbxl_meta_input_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_meta(bad_dir)


def test_load_ptbxl_meta_missing_ptbxl_database_csv(tmp_path):
    # Create valid directory but no ptbxl_database.csv file
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scp_statements.csv").write_text("diagnostic\n1")  # dummy content
    with pytest.raises(FileNotFoundError, match=r"No such file or directory"):
        load_ptbxl_meta(tmp_path)


def test_load_ptbxl_meta_missing_scp_statements_csv(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ptbxl_database.csv").write_text(
        "ecg_id,scp_codes\n1000,\"{'NORM': 1}\""
    )
    with pytest.raises(FileNotFoundError, match=r"No such file or directory"):
        load_ptbxl_meta(tmp_path)


def test_load_ptbxl_meta_success_with_mock(tmp_path):
    # Create sample ptbxl_database.csv
    ptb_csv = tmp_path / "ptbxl_database.csv"
    ptb_csv.write_text("ecg_id,scp_codes\n1001,\"{'NORM': 1.0}\"")

    # Create sample scp_statements.csv
    scp_csv = tmp_path / "scp_statements.csv"
    scp_csv.write_text("scp_code,diagnostic\nNORM,1")

    with patch("ecg_cnn.data.data_utils.aggregate_diagnostic") as mock_agg:
        mock_agg.return_value = ["NORM"]
        df = load_ptbxl_meta(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert "diagnostic_superclass" in df.columns
        assert df.loc[1001, "diagnostic_superclass"] == ["NORM"]
        mock_agg.assert_called_once()


def test_load_ptbxl_meta_success_no_mock(tmp_path):
    # --- Create ptbxl_database.csv with a valid scp_codes dictionary ---
    ptb_csv = tmp_path / "ptbxl_database.csv"
    ptb_csv.write_text("ecg_id,scp_codes\n1001,\"{'NORM': 1.0}\"")

    # --- Create scp_statements.csv with NORM as a diagnostic SCP code ---
    scp_csv = tmp_path / "scp_statements.csv"
    scp_csv.write_text("scp_code,diagnostic,diagnostic_class\nNORM,1,NORM")

    # --- Call the actual function (no mocking) ---
    df = load_ptbxl_meta(tmp_path)

    # --- Validate result ---
    assert isinstance(df, pd.DataFrame)
    assert "diagnostic_superclass" in df.columns
    assert df.loc[1001, "diagnostic_superclass"] == ["NORM"]


def test_load_ptbxl_meta_invalid_scp_codes_format(tmp_path):
    (tmp_path / "ptbxl_database.csv").write_text('ecg_id,scp_codes\n1002,"not_a_dict"')
    (tmp_path / "scp_statements.csv").write_text("diagnostic\n1")

    with pytest.raises(ValueError, match=r"^malformed node or string"):
        load_ptbxl_meta(tmp_path)


# ------------------------------------------------------------------------------
# def load_ptbxl_sample(sample_dir, ptb_path):
# ------------------------------------------------------------------------------


def test_load_ptbxl_sample_input_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    ptb_path = tmp_path / "samples"
    ptb_path.mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_sample(bad_dir, ptb_path)


def test_load_ptbxl_sample_ptb_path_not_a_directory(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)
    bad_dir = tmp_path / "does_not_exist"
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=bad_dir)


def test_load_ptbxl_sample_from_ids_csv(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    ecg_ids = [12345]
    pd.DataFrame({"ecg_id": ecg_ids}).to_csv(sample_dir / "sample_ids.csv", index=False)

    csv_path = sample_dir / "12345_lr_100hz.csv"
    csv_path.write_text("\n".join(",".join(["0"] * 12) for _ in range(1000)) + "\n")

    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    dummy_meta = pd.DataFrame(
        {
            "ecg_id": ecg_ids,
            "filename_lr": ["00000/12345_lr"],
            "scp_codes": [{"NORM": 1.0}],
            "diagnostic": [1],
        }
    ).set_index("ecg_id")

    dummy_X = np.zeros((1, 12, 1000))
    dummy_y = np.array([1])

    with (
        patch(
            "ecg_cnn.data.data_utils.build_full_X_y",
            return_value=(dummy_X, dummy_y, dummy_meta),
        ),
        patch("ecg_cnn.data.data_utils.load_ptbxl_meta", return_value=dummy_meta),
        patch("wfdb.rdsamp", return_value=(np.zeros((1000, 12)), {})),
    ):
        X, y, meta = load_ptbxl_sample(sample_dir, ptb_dir)

    y = np.array(y)
    assert X.shape == (1, 12, 1000)
    assert y.shape == (1,)
    assert meta.shape[0] == 1


def test_load_ptbxl_sample_from_filename_parsing(tmp_path):
    # Prepare sample directory with dummy files
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)

    # Must match the loader’s expected pattern and be numeric-only
    csv_path = sample_dir / "00123_lr_100hz.csv"  # <-- was 00123_lr.csv
    csv_path.write_text("\n".join(",".join(["0"] * 12) for _ in range(1000)) + "\n")

    (sample_dir / "ignore.txt").write_text("skip this")  # invalid
    (sample_dir / "badid_csv.csv").write_text("skip this too")  # malformed ID

    ptb_path = tmp_path / "ptbxl"
    ptb_path.mkdir(parents=True)

    dummy_meta = pd.DataFrame(
        {
            "ecg_id": [123],
            "filename_lr": ["00000/123_lr"],
            "scp_codes": [{"MI": 1.0}],
            "diagnostic": [1],
        }
    ).set_index("ecg_id")

    dummy_X = np.zeros((1, 12, 1000))
    dummy_y = np.array([1])

    with (
        patch(
            "ecg_cnn.data.data_utils.build_full_X_y",
            return_value=(dummy_X, dummy_y, dummy_meta),
        ),
        patch("ecg_cnn.data.data_utils.load_ptbxl_meta", return_value=dummy_meta),
        patch("wfdb.rdsamp", return_value=(np.zeros((1000, 12)), {})),
    ):
        X, y, meta = load_ptbxl_sample(sample_dir, ptb_path)

    y = np.array(y)
    assert X.shape == (1, 12, 1000)
    assert y.shape == (1,)
    assert meta.shape[0] == 1


def test_load_ptbxl_sample_sample_dir_bad_type_triggers_typeerror():
    # sample_dir invalid type -> hits the FIRST else (raises TypeError) before anything else
    with pytest.raises(TypeError, match=r"^sample_dir must be str\|Path\|None"):
        load_ptbxl_sample(sample_dir=123, ptb_path=None)


def test_load_ptbxl_sample_ptb_path_string_hits_second_if_and_then_notadir(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=str(tmp_path / "ptb_dne"))


def test_load_ptbxl_sample_normalizes_ptb_path_string(tmp_path):
    # Valid sample_dir so we get past its checks
    sample_dir = tmp_path / "sample_ok2"
    sample_dir.mkdir()

    # Provide ptb_path as STRING to hit the `elif isinstance(ptb_path, (str, Path))` branch
    ptb_path_str = str(tmp_path / "ptb_as_str")
    Path(ptb_path_str).mkdir(parents=True, exist_ok=True)

    # Will fail later due to missing expected files — that’s fine for coverage.
    with pytest.raises(
        FileNotFoundError, match=r"^\[Errno 2\] No such file or directory"
    ):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=ptb_path_str)


def test_load_ptbxl_sample_uses_default_sample_dir_when_none(tmp_path, patch_paths):
    # Ensure the default sample directory exists
    (tmp_path / "data" / "sample").mkdir(parents=True, exist_ok=True)
    # Ensure default PTB dir exists as directory
    (tmp_path / "ptbxl").mkdir(parents=True, exist_ok=True)

    # We only care that the branch executes; the function will fail later when it
    # can't find the sample CSVs. That’s fine for coverage.
    with pytest.raises(FileNotFoundError, match=r"No such file or directory"):
        load_ptbxl_sample(sample_dir=None, ptb_path=None)


def test_load_ptbxl_sample_rejects_bad_ptb_path_type(tmp_path):
    # Valid sample_dir so we reach the ptb_path type check
    sample_dir = tmp_path / "sample_ok3"
    sample_dir.mkdir()

    # Invalid ptb_path type -> hits the final 'else' and raises TypeError
    with pytest.raises(TypeError, match=r"^ptb_path must be str\|Path\|None"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=[])


def test_load_ptbxl_sample_sample_only_triggers_fallback(tmp_path):
    # Create minimal CSV with 12 columns so it looks like ECG
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    csv_path = sample_dir / "00001_lr_100hz.csv"
    csv_path.write_text("\n".join(",".join(["0"] * 12) for _ in range(5)))

    # Call with sample_only=True forces line 452 path
    X, y, meta = load_ptbxl_sample(
        sample_dir=sample_dir, ptb_path=None, sample_only=True
    )
    assert X.shape[1] == 12
    assert meta.shape[0] == X.shape[0]
    assert y.dtype == np.int64


def test_load_ptbxl_sample_fallback_loader_reads_csvs(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # Write two valid 12-lead CSVs
    for i in range(2):
        f = sample_dir / f"{i:05d}_lr_100hz.csv"
        f.write_text("\n".join(",".join(["0"] * 12) for _ in range(5)))

    X, y, meta = load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "notadir")
    assert X.shape[0] == 2
    assert len(y) == 2
    assert "diagnostic_superclass" in meta.columns


def test_load_ptbxl_sample_raises_if_ptb_path_invalid_and_not_sample_mode(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "missing_dir")


def test_load_ptbxl_sample_load_ptbxl_sample_reads_with_wfdb(monkeypatch, tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    csv_path = sample_dir / "00001_lr_100hz.csv"
    csv_path.write_text("\n".join(",".join(["0"] * 12) for _ in range(5)))

    ptb_path = tmp_path / "ptbxl"
    ptb_path.mkdir()

    # Fake metadata with one entry
    meta = pd.DataFrame(
        {"ecg_id": [1], "filename_lr": ["00001_lr"], "scp_codes": [{"NORM": 1.0}]}
    ).set_index("ecg_id")

    # Patch load_ptbxl_meta and wfdb.rdsamp
    from ecg_cnn import data as data_pkg

    monkeypatch.setattr(
        data_pkg.data_utils, "load_ptbxl_meta", lambda _: meta, raising=False
    )
    monkeypatch.setattr(
        "wfdb.rdsamp", lambda path: (np.zeros((5, 12)), {}), raising=False
    )
    monkeypatch.setattr(
        data_pkg.data_utils, "raw_to_five_class", lambda scp: "NORM", raising=False
    )

    X, y, meta_out = load_ptbxl_sample(sample_dir=sample_dir, ptb_path=ptb_path)
    assert X.shape[1] == 12
    assert y.shape[0] == 1
    assert "diagnostic_superclass" in meta_out.columns


def test_load_ptbxl_sample_fallback_loader_hits_elif_and_else_branches(tmp_path):
    """
    Covers :
      - one file with shape (12, T) hits `elif arr.shape[0] == 12: pass`
      - one file with shape (7, 9) hits `else: continue` (skipped)
    """
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    good = sample_dir / "00001_lr_100hz.csv"
    bad = sample_dir / "00002_lr_100hz.csv"
    _write_csv(good, rows=12, cols=5)  # (12, T) -> uses `elif` branch
    _write_csv(bad, rows=7, cols=9)  # neither dimension 12 -> `continue`

    # Force fallback mini-loader (ptb_path not a dir)
    X, y, meta = load_ptbxl_sample(
        sample_dir=sample_dir, ptb_path=tmp_path / "missing_dir"
    )
    assert X.shape[0] == 1  # bad one skipped
    assert X.shape[1] == 12  # channel dimension preserved
    assert len(y) == 1
    assert meta.shape[0] == 1


def test_load_ptbxl_sample_fallback_loader_bad_stem_triggers_id_fallback(tmp_path):
    """
    Covers: filename stem not int → except → ecg_id = i+1
    """
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # non-numeric stem before first underscore
    weird = sample_dir / "abc_lr_100hz.csv"
    _write_csv(weird, rows=12, cols=5)

    X, y, meta = load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "nowhere")
    # Expect ecg_id synthesized as 1 (i starts at 0)
    assert int(meta["ecg_id"].iloc[0]) == 1


def test_load_ptbxl_sample_fallback_loader_raises_when_no_valid_12lead_csvs(tmp_path):
    """
    Covers 494: all CSVs skipped -> X_list empty -> ValueError
    """
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # Two files, neither with a 12-sized dimension
    _write_csv(sample_dir / "00003_lr_100hz.csv", rows=5, cols=5)
    _write_csv(sample_dir / "00004_lr_100hz.csv", rows=8, cols=10)

    with pytest.raises(ValueError, match=r"No valid 12-lead CSVs"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "notadir")


def test_load_ptbxl_sample_not_sample_mode_invalid_ptb_path_raises(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)  # empty → not sample mode
    with pytest.raises(NotADirectoryError, match=r"^Input not a directory"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "missing_dir")


def test_load_ptbxl_sample_synth_meta_when_load_meta_fails_in_sample_mode(
    monkeypatch, tmp_path
):
    """
    Covers 551: load_ptbxl_meta raises, sample_mode True -> synthesize meta DataFrame.
    """
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # Provide sample_ids.csv to enter sample_mode branch
    pd.DataFrame({"ecg_id": [123]}).to_csv(sample_dir / "sample_ids.csv", index=False)
    # Also provide the per-ID CSV used later in sample_mode loop
    id_csv = sample_dir / "00123_lr_100hz.csv"
    _write_csv(id_csv, rows=12, cols=5)

    # Make load_ptbxl_meta fail to force the except path
    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.load_ptbxl_meta",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    # ptb_path can be anything; sample_mode True sidesteps the check
    X, y, meta = load_ptbxl_sample(
        sample_dir=sample_dir, ptb_path=tmp_path / "anything"
    )
    assert X.shape[0] == 1
    assert "diagnostic_superclass" in meta.columns  # added later if missing


def test_load_ptbxl_sample_wfdb_path_is_used_in_non_sample_mode(monkeypatch, tmp_path):
    """
    Covers: non-sample mode (no CSVs), valid ptb_path, use wfdb.rdsamp.
    """
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()  # empty => not sample mode

    ptb_path = tmp_path / "ptbxl"
    ptb_path.mkdir()

    # Fake metadata with one entry; filename_lr joined under ptb_path
    meta = pd.DataFrame(
        {"ecg_id": [1], "filename_lr": ["00001_lr"], "scp_codes": [{"NORM": 1.0}]}
    ).set_index("ecg_id")

    # Patch meta loader and wfdb.rdsamp, plus raw_to_five_class
    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.load_ptbxl_meta", lambda *_: meta, raising=False
    )
    monkeypatch.setattr("wfdb.rdsamp", lambda p: (np.zeros((7, 12)), {}), raising=False)
    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.raw_to_five_class", lambda scp: "NORM", raising=False
    )

    X, y, meta_out = load_ptbxl_sample(sample_dir=sample_dir, ptb_path=ptb_path)
    assert X.shape == (1, 12, 7)  # (T,12) transposed to (12,T)
    assert y.shape == (1,)
    assert meta_out.shape[0] == 1


def test_load_ptbxl_sample_fallback_raises_when_csv_suffix_but_no_files(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # Create a directory with .csv suffix to trigger sample_mode=True
    (sample_dir / "fake.csv").mkdir()
    # Missing PTB dir forces fallback branch
    with pytest.raises(FileNotFoundError, match=r"No CSVs found in sample_dir"):
        load_ptbxl_sample(sample_dir=sample_dir, ptb_path=tmp_path / "missing_root")


def test_load_ptbxl_sample_synthesize_meta_when_meta_load_fails_in_sample_mode(
    monkeypatch, tmp_path
):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    # sample_mode=True via sample_ids.csv
    ecg_id = 123
    pd.DataFrame({"ecg_id": [ecg_id]}).to_csv(
        sample_dir / "sample_ids.csv", index=False
    )
    # per-ID signal CSV (12 cols, a few rows; comma-separated)
    sig = "\n".join(",".join(["0"] * 12) for _ in range(5)) + "\n"
    (sample_dir / f"{ecg_id:05d}_lr_100hz.csv").write_text(sig)

    # ptb_path must exist so we hit try/except (not the earlier fallback/return)
    ptb_path = tmp_path / "ptbxl"
    ptb_path.mkdir()

    # Force load_ptbxl_meta to fail -> triggers synth at line 761
    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.load_ptbxl_meta",
        lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        raising=False,
    )

    X, y, meta = load_ptbxl_sample(sample_dir=sample_dir, ptb_path=ptb_path)
    assert X.shape[1] == 12 and X.shape[0] == 1
    assert y.shape == (1,)
    assert "diagnostic_superclass" in meta.columns


def test_load_ptbxl_sample_loader_rounds_out_all_five_classes(tmp_path):
    """
    When fewer than 5 labels are present, it round-robins missing class ids
    into the first few samples so that all 5 classes (0..4) appear.
    """
    sample_dir: Path = tmp_path / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 simple 12-lead CSVs (shape (T,12)) so not all 5 classes are present
    # The loader will transpose if needed and fabricate labels in a 5-class cycle.
    T = 20
    leads = 12
    for i in range(1, 4):  # only 3 CSVs -> <5 classes present initially
        arr = np.arange(T * leads, dtype=float).reshape(T, leads)
        np.savetxt(sample_dir / f"{i:05d}_toy.csv", arr, delimiter=",", fmt="%.1f")

    # Force the 'ptb_path missing' branch to trigger sample-mode fallback:
    missing_ptb = tmp_path / "ptb_root_does_not_exist"

    X, y_int, meta = load_ptbxl_sample(
        sample_dir=sample_dir,
        ptb_path=missing_ptb,  # not a dir -> triggers the code block with branch 516
        sample_only=True,
    )

    # Sanity: we loaded our 3 files
    assert X.shape[0] == 3
    assert len(y_int) == 3
    assert len(meta) == 3

    # After the round-robin step, labels remain in 0..4 and
    # we should have as many distinct labels as samples (max diversity for N<5).
    labels = set(y_int.tolist())
    assert labels.issubset(set(range(5)))
    assert len(labels) == len(y_int)  # e.g., for N=3 you’ll see 3 distinct labels


def test_sample_loader_no_round_robin_when_all_five_classes_present(tmp_path):
    """
    With 5 CSVs, the fabricated labels cycle 0..4, so the round-robin block is
    skipped.
    """
    sample_dir = tmp_path / "sample_all5"
    sample_dir.mkdir(parents=True, exist_ok=True)

    T, leads = 20, 12
    # Create 5 CSVs -> label_cycle will yield 0,1,2,3,4 exactly once
    for i in range(1, 6):
        arr = np.arange(T * leads, dtype=float).reshape(T, leads)
        np.savetxt(sample_dir / f"{i:05d}_toy.csv", arr, delimiter=",", fmt="%.1f")

    missing_ptb = tmp_path / "ptb_root_does_not_exist"
    X, y_int, meta = load_ptbxl_sample(
        sample_dir=sample_dir,
        ptb_path=missing_ptb,  # force sample-mode block
        sample_only=True,
    )

    assert X.shape[0] == 5
    assert len(y_int) == 5

    # All five classes already present -> condition at 516 is FALSE -> no changes needed
    assert set(y_int.tolist()) == set(range(5))


def test_load_ptbxl_sample_adds_missing_superclass(tmp_path):
    """
    Covers: data_utils.load_ptbxl_sample line 593
    Case where 'diagnostic_superclass' is missing in sample_meta,
    so the function must insert it before returning.
    """
    # --- make a fake sample_dir with a minimal CSV (valid 12-lead shape) ---
    csv_path = tmp_path / "00001_lr_100hz.csv"
    # Shape (100, 12) -> will transpose to (12, 100)
    df = pd.DataFrame(np.ones((100, 12)))
    df.to_csv(csv_path, index=False, header=False)

    # --- also create a dummy ptb_path that isn't a dir, so sample_mode triggers ---
    ptb_path = tmp_path / "not_a_dir"

    # Run the function: forces sample_mode=True, sample_meta initially has no diagnostic_superclass
    X, y, meta = load_ptbxl_sample(sample_dir=tmp_path, ptb_path=ptb_path)

    # Assertions
    assert isinstance(X, np.ndarray) and X.ndim == 3  # (N, 12, T)
    assert isinstance(y, np.ndarray) and y.dtype == np.int64
    # The key check: diagnostic_superclass column must have been added
    assert "diagnostic_superclass" in meta.columns
    assert len(meta) == len(X) == len(y)


def test_load_ptbxl_sample_adds_superclass_when_missing_ptb_meta(tmp_path, monkeypatch):
    """
    Covers True branch at line ~593:
    sample_meta lacks 'diagnostic_superclass' -> function must add it.
    """
    # sample_dir with one valid ECG CSV (shape (T,12) so it transposes to (12,T))
    csv_path = tmp_path / "00001_lr_100hz.csv"
    pd.DataFrame(np.ones((100, 12))).to_csv(csv_path, index=False, header=False)

    # ptb_path must be a directory so we go past the early-return sample-only path
    ptb_path = tmp_path / "ptb"
    ptb_path.mkdir()

    # Return PTB meta WITHOUT 'diagnostic_superclass' to trigger the add
    def _fake_meta(_):
        return pd.DataFrame(
            {
                "ecg_id": [1],
                "filename_lr": [""],
                "scp_codes": [{}],
            }
        ).set_index("ecg_id")

    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.load_ptbxl_meta", _fake_meta, raising=False
    )

    X, y, meta = load_ptbxl_sample(
        sample_dir=tmp_path, ptb_path=ptb_path, sample_only=False
    )

    assert isinstance(X, np.ndarray) and X.ndim == 3
    assert isinstance(y, np.ndarray) and y.dtype == np.int64
    # Key assertion: column was added
    assert "diagnostic_superclass" in meta.columns
    assert len(meta) == len(X) == len(y)


def test_load_ptbxl_sample_keeps_existing_superclass_if_present(tmp_path, monkeypatch):
    """
    Covers False branch at line ~593:
    sample_meta already has 'diagnostic_superclass' -> function should NOT add it.
    """
    # sample_dir with one valid ECG CSV
    csv_path = tmp_path / "00002_lr_100hz.csv"
    pd.DataFrame(np.ones((120, 12))).to_csv(csv_path, index=False, header=False)

    # ptb_path is a directory so we run the main path (no early return)
    ptb_path = tmp_path / "ptb_full"
    ptb_path.mkdir()

    # PTB meta WITH 'diagnostic_superclass' present
    def _fake_meta(_):
        return pd.DataFrame(
            {
                "ecg_id": [2],
                "filename_lr": [""],
                "scp_codes": [{}],
                "diagnostic_superclass": ["NORM"],
            }
        ).set_index("ecg_id")

    monkeypatch.setattr(
        "ecg_cnn.data.data_utils.load_ptbxl_meta", _fake_meta, raising=False
    )

    X, y, meta = load_ptbxl_sample(
        sample_dir=tmp_path, ptb_path=ptb_path, sample_only=False
    )

    assert isinstance(X, np.ndarray) and X.ndim == 3
    assert isinstance(y, np.ndarray) and y.dtype == np.int64
    # Key assertion: column was already there and remains
    assert "diagnostic_superclass" in meta.columns
    assert len(meta) == len(X) == len(y)


# ------------------------------------------------------------------------------
# def load_ptbxl_full(data_dir, subsample_frac, sampling_rate=100):
# ------------------------------------------------------------------------------
def test_load_ptbxl_full_data_dir_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    subsample_frac = 0.5
    sampling_rate = 100
    with pytest.raises(FileNotFoundError, match=r"^PTB-XL data_dir missing:"):
        load_ptbxl_full(bad_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sample_frac_zero(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 0.0
    sampling_rate = 100
    with pytest.raises(ValueError, match=r"^subsample_frac must be in \(0\.0, 1\.0\]"):
        load_ptbxl_full(data_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sample_frac_over_one(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 1.01
    sampling_rate = 100
    with pytest.raises(ValueError, match=r"^subsample_frac must be in \(0\.0, 1\.0\]"):
        load_ptbxl_full(data_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sampling_rate_invalid(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 1.0
    sampling_rate = 666
    with pytest.raises(ValueError, match=r"^sampling_rate must be 100 or 500"):
        load_ptbxl_full(data_dir, subsample_frac, sampling_rate)


@patch("ecg_cnn.data.data_utils.wfdb.rdsamp")
@patch("ecg_cnn.data.data_utils.raw_to_five_class", return_value="MI")
def test_load_ptbxl_full_100hz_full_sample(mock_map, mock_rdsamp, tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)

    # Create dummy metadata
    df = pd.DataFrame(
        {
            "ecg_id": [123],
            "filename_lr": ["records100/00000/00123_lr"],
            "filename_hr": ["records500/00000/00123_hr"],
            "scp_codes": [{"MI": 1.0}],
        }
    ).set_index("ecg_id")
    df.to_csv(data_dir / "ptbxl_database.csv")

    # Fake waveform file structure (ensure .hea/.dat exist if code checks for them)
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00123_lr.hea").touch()
    (wfdb_dir / "00123_lr.dat").touch()

    # Create corresponding scp_statements.csv
    create_dummy_scp_statements(data_dir)

    # Mock waveform return: shape (1000, 12) -> transposed to (12, 1000)
    mock_rdsamp.return_value = (np.random.randn(1000, 12), {})

    # Call function under test
    X, y, meta = load_ptbxl_full(data_dir, subsample_frac=1.0, sampling_rate=100)

    assert X.shape == (1, 12, 1000)
    assert y == ["MI"]
    assert len(meta) == 1


@patch("ecg_cnn.data.data_utils.wfdb.rdsamp")
@patch("ecg_cnn.data.data_utils.raw_to_five_class", return_value="MI")
def test_load_ptbxl_full_500hz_full_sample(mock_map, mock_rdsamp, tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)

    # Create minimal ptbxl_database.csv with 1 ECG entry
    df = pd.DataFrame(
        {
            "ecg_id": [456],
            "filename_lr": ["records100/00000/00456_lr"],
            "filename_hr": ["records500/00000/00456_hr"],
            "scp_codes": [{"MI": 1.0}],
        }
    ).set_index("ecg_id")
    df.to_csv(data_dir / "ptbxl_database.csv")

    # Create dummy waveform file path (ensure .hea/.dat exist if code checks)
    wfdb_dir = data_dir / "records500" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00456_hr.hea").touch()
    (wfdb_dir / "00456_hr.dat").touch()

    # Create corresponding scp_statements.csv
    create_dummy_scp_statements(data_dir)

    # Mock waveform return: shape (2000, 12) -> transposed to (12, 2000)
    mock_rdsamp.return_value = (np.random.randn(2000, 12), {})

    # Call function under test
    X, y, meta = load_ptbxl_full(data_dir, subsample_frac=1.0, sampling_rate=500)

    assert X.shape == (1, 12, 2000)
    assert y == ["MI"]
    assert len(meta) == 1


@patch("ecg_cnn.data.data_utils.wfdb.rdsamp")
@patch("ecg_cnn.data.data_utils.raw_to_five_class", return_value="NORM")
def test_load_ptbxl_full_subsample(mock_map, mock_rdsamp, tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)

    ids = list(range(1000, 1005))
    df = pd.DataFrame(
        {
            "ecg_id": ids,
            "filename_lr": [f"records100/00000/{i:05d}_lr" for i in ids],
            "filename_hr": [f"records500/00000/{i:05d}_hr" for i in ids],
            "scp_codes": [{"NORM": 1.0} for _ in ids],
        }
    ).set_index("ecg_id")
    df.to_csv(data_dir / "ptbxl_database.csv")

    # Create dummy waveform file paths (ensure .hea/.dat exist if code checks)
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    for i in ids:
        (wfdb_dir / f"{i:05d}_lr.hea").touch()
        (wfdb_dir / f"{i:05d}_lr.dat").touch()

    # Create corresponding scp_statements.csv
    create_dummy_scp_statements(data_dir)

    # Mock waveform return: shape (500, 12) -> transposed to (12, 500)
    mock_rdsamp.return_value = (np.random.randn(500, 12), {})

    # Call function under test
    X, y, meta = load_ptbxl_full(data_dir, subsample_frac=0.4, sampling_rate=100)

    assert 1 <= len(X) <= 4  # due to rounding of 5 * 0.4 = 2
    assert all(label == "NORM" for label in y)
    assert meta.shape[0] == len(X)


@patch("ecg_cnn.data.data_utils.wfdb.rdsamp")
@patch("ecg_cnn.data.data_utils.raw_to_five_class", return_value="Unknown")
def test_load_ptbxl_full_returns_unknown_label(mock_map, mock_rdsamp, tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)

    # Create minimal ptbxl_database.csv with 1 ECG entry
    df = pd.DataFrame(
        {
            "ecg_id": [789],
            "filename_lr": ["records100/00000/00789_lr"],
            "filename_hr": ["records500/00000/00789_hr"],
            "scp_codes": [{"ALIEN": 1.0}],
        }
    ).set_index("ecg_id")
    df.to_csv(data_dir / "ptbxl_database.csv")

    # Create dummy waveform file path (ensure .hea/.dat exist if code checks)
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00789_lr.hea").touch()
    (wfdb_dir / "00789_lr.dat").touch()

    # Create corresponding scp_statements.csv
    create_dummy_scp_statements(data_dir)

    # Mock waveform return: shape (500, 12) -> transposed to (12, 500)
    mock_rdsamp.return_value = (np.random.randn(500, 12), {})

    # Call function under test
    X, y, meta = load_ptbxl_full(data_dir, subsample_frac=1.0, sampling_rate=100)

    assert X.shape == (1, 12, 500)
    assert y == ["Unknown"]
    assert len(meta) == 1


@patch(
    "ecg_cnn.data.data_utils.wfdb.rdsamp", side_effect=RuntimeError("mock read error")
)
def test_load_ptbxl_full_handles_rdsamp_failure(mock_rdsamp, tmp_path):
    # Create minimal record path
    data_dir = tmp_path
    records_dir = data_dir / "records100" / "10000"
    records_dir.mkdir(parents=True)
    (records_dir / "dummy.hea").write_text("")  # ensure header exists if code checks
    (records_dir / "dummy.dat").write_text("")  # placeholder signal file

    # Create minimal ptbxl_database.csv
    meta = pd.DataFrame(
        {
            "ecg_id": [1],
            "filename_lr": ["records100/10000/dummy"],
            "scp_codes": ['{"NORM": 1}'],
        }
    )
    meta_path = data_dir / "ptbxl_database.csv"
    meta.to_csv(meta_path, index=False)

    # Create minimal scp_statements.csv (required by load_ptbxl_meta)
    scp_df = pd.DataFrame(
        {
            "diagnostic_class": ["NORM"],
            "diagnostic": [1],
        },
        index=["NORM"],
    )
    scp_path = data_dir / "scp_statements.csv"
    scp_df.to_csv(scp_path)

    # Run function — it should raise an error due to all records failing
    with pytest.raises(ValueError, match=r"^No valid records were loaded\."):
        load_ptbxl_full(data_dir=str(data_dir), subsample_frac=1.0)


def test_load_ptbxl_full_data_dir_none_raises():
    with pytest.raises(FileNotFoundError, match=r"^PTB-XL data_dir missing: None"):
        load_ptbxl_full(
            data_dir=None,
            subsample_frac=0.5,
            sampling_rate=100,
        )


# ------------------------------------------------------------------------------
# def raw_to_five_class(scp_entry) -> str:
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_val, expected",
    [
        # Valid string with highest-priority label
        ("{'MI': 1.0, 'SR': 0.0}", "MI"),
        # Valid string with known mapping
        ("{'AFLT': 1.0}", "STTC"),
        # Valid string with unknown label
        ("{'XYZ': 1.0}", "Unknown"),
        # Valid dictionary input
        ({"LVH": 1.0}, "HYP"),
        # Not a string or a dictionary
        (42, "Unknown"),
        # Can't be parsed
        ("bad json", "Unknown"),
        # Empty dictionary
        ({}, "Unknown"),
        # Empty stringified dictionary
        ("{}", "Unknown"),
    ],
)
def test_raw_to_five_class_cases(input_val, expected):
    result = raw_to_five_class(input_val)
    assert result == expected
