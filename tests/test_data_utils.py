import numpy as np
import os
import pandas as pd
import pytest
import torch
import wfdb

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
)


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------


def write_mock_ptb_record(dir_path, ecg_id):
    """Creates fake WFDB record files for given ecg_id."""
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

    with pytest.raises(FileNotFoundError, match="Metadata CSV not found"):
        build_full_X_y(str(meta), str(scp), str(ptb_dir))


def test_build_full_X_y_missing_scp_csv(tmp_path):
    """Should raise FileNotFoundError if scp CSV is missing."""
    meta = tmp_path / "meta.csv"
    meta.write_text("dummy\n")

    scp = tmp_path / "scp.csv"  # intentionally not created

    ptb_dir = tmp_path / "ptb"
    ptb_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="SCP statements CSV not found"):
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

    with pytest.raises(NotADirectoryError, match="PTB-XL root directory not found"):
        build_full_X_y(meta, scp, bad_dir)


def test_build_full_X_y_smoke(tmp_path):
    # Create minimal meta CSV
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "ecg_id,filename_lr,scp_codes\n"
        "10001,records100/10000/10001_lr,\"['NORM']\"\n"
    )

    # Create dummy scp CSV
    scp = tmp_path / "scp.csv"
    scp.write_text("dummy,desc\n")

    # Use real PTB-XL data dir if available (or copy one real file into tmp_path)
    ptb_dir = Path("data/ptbxl/physionet.org/files/ptb-xl/1.0.3")  # adjust if needed
    if not ptb_dir.exists():
        pytest.skip("PTB-XL data directory not found")

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
    assert y[0] == 3  # assuming LABEL2IDX['NORM'] == 3


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

    with pytest.raises(ValueError, match="No valid records"):
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

    with pytest.raises(ValueError, match="No valid records"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_missing_filename_lr(tmp_path):
    """Should skip rows with missing filename_lr (empty)."""
    ptb_dir = tmp_path / "ptbxl"
    ptb_dir.mkdir(parents=True)

    meta_csv = tmp_path / "meta.csv"
    meta_csv.write_text("ecg_id,filename_lr,scp_codes\n123,,\"['NORM']\"\n")

    scp_csv = tmp_path / "scp.csv"
    scp_csv.write_text("code,description\nNORM,Normal\n")

    with pytest.raises(ValueError, match="No valid records were loaded"):
        build_full_X_y(meta_csv, scp_csv, ptb_dir)


def test_build_full_X_y_missing_filename_lr(tmp_path):
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

    with pytest.raises(ValueError, match="No valid records"):
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
    with pytest.raises(TypeError, match="label_list must be a list, set, or tuple"):
        select_primary_label("MI")  # not a list-like structure


def test_select_primary_label_type_error_on_non_string_elements():
    with pytest.raises(TypeError, match="All elements in label_list must be strings"):
        select_primary_label(["MI", 123])


# ------------------------------------------------------------------------------
# def aggregate_diagnostic(codes, agg_df, restrict=True):
# ------------------------------------------------------------------------------


def test_aggregate_diagnostic_codes_not_a_dictionary():
    dummy_df = pd.DataFrame(columns=["scp_code", "diagnostic", "diagnostic_class"])
    with pytest.raises(ValueError, match="must be a dictionary"):
        aggregate_diagnostic('{"NORM": 1.0, "MI": 0.0}', "not_a_df", restrict=True)


def test_aggregate_diagnostic_not_dataframe():
    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
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


def test_aggregate_diagnostic_restrict_false_unnusual_code():
    dummy_df = pd.DataFrame(
        {
            "scp_code": ["FAKE"],
            "diagnostic": [1],  # Diagnostic, but class is not in target list
            "diagnostic_class": ["ALIEN"],
        }
    ).set_index("scp_code")
    result = aggregate_diagnostic({"FAKE": 1.0}, dummy_df, restrict=False)
    assert result == ["ALIEN"]


# ------------------------------------------------------------------------------
# def load_ptbxl_meta(ptb_path):
# ------------------------------------------------------------------------------


def test_load_ptbxl_meta_input_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    with pytest.raises(NotADirectoryError, match="Input not a directory"):
        load_ptbxl_meta(bad_dir)


def test_load_ptbxl_meta_missing_ptbxl_database_csv(tmp_path):
    # Create valid directory but no ptbxl_database.csv file
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scp_statements.csv").write_text("diagnostic\n1")  # dummy content
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        load_ptbxl_meta(tmp_path)


def test_load_ptbxl_meta_missing_scp_statements_csv(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ptbxl_database.csv").write_text(
        "ecg_id,scp_codes\n1000,\"{'NORM': 1}\""
    )
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
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

    with pytest.raises(ValueError, match="malformed node or string"):
        load_ptbxl_meta(tmp_path)


# ------------------------------------------------------------------------------
# def load_ptbxl_sample(sample_dir, ptb_path):
# ------------------------------------------------------------------------------


def test_load_ptbxl_sample_input_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    ptb_path = tmp_path / "samples"
    ptb_path.mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match="Input not a directory"):
        load_ptbxl_sample(bad_dir, ptb_path)


def test_load_ptbxl_sample_pbt_path_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)
    with pytest.raises(NotADirectoryError, match="Input not a directory"):
        load_ptbxl_sample(sample_dir, bad_dir)


def test_load_ptbxl_sample_from_ids_csv(tmp_path):

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir(parents=True)
    ecg_ids = [12345]
    pd.DataFrame({"ecg_id": ecg_ids}).to_csv(sample_dir / "sample_ids.csv", index=False)
    (sample_dir / "12345_lr.csv").write_text("dummy")

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

    with patch(
        "ecg_cnn.data.data_utils.build_full_X_y",
        return_value=(dummy_X, dummy_y, dummy_meta),
    ), patch("ecg_cnn.data.data_utils.load_ptbxl_meta", return_value=dummy_meta), patch(
        "wfdb.rdsamp", return_value=(np.zeros((1000, 12)), {})
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
    (sample_dir / "00123_lr.csv").write_text("dummy")  # valid
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

    with patch(
        "ecg_cnn.data.data_utils.build_full_X_y",
        return_value=(dummy_X, dummy_y, dummy_meta),
    ), patch("ecg_cnn.data.data_utils.load_ptbxl_meta", return_value=dummy_meta), patch(
        "wfdb.rdsamp", return_value=(np.zeros((1000, 12)), {})
    ):

        X, y, meta = load_ptbxl_sample(sample_dir, ptb_path)

    y = np.array(y)
    assert X.shape == (1, 12, 1000)
    assert y.shape == (1,)
    assert meta.shape[0] == 1


# ------------------------------------------------------------------------------
# def load_ptbxl_full(data_dir, subsample_frac, sampling_rate=100):
# ------------------------------------------------------------------------------
def test_load_ptbxl_full_data_dir_not_a_directory(tmp_path):
    bad_dir = tmp_path / "nonexistent" / "dir"
    subsample_frac = 0.5
    sampling_rate = 100
    with pytest.raises(NotADirectoryError, match="Invalid data directory"):
        load_ptbxl_full(bad_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sample_frac_zero(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 0.0
    sampling_rate = 100
    with pytest.raises(ValueError, match=r"subsample_frac must be in \(0\.0, 1\.0\]"):
        load_ptbxl_full(data_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sample_frac_over_one(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 1.01
    sampling_rate = 100
    with pytest.raises(ValueError, match=r"subsample_frac must be in \(0\.0, 1\.0\]"):
        load_ptbxl_full(data_dir, subsample_frac, sampling_rate)


def test_load_ptbxl_full_sampling_rate_invalid(tmp_path):
    data_dir = tmp_path / "ptbxl"
    data_dir.mkdir(parents=True)
    subsample_frac = 1.0
    sampling_rate = 666
    with pytest.raises(ValueError, match=r"sampling_rate must be 100 or 500"):
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

    # Fake waveform file structure
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00123_lr").touch()

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

    # Create dummy waveform file path
    wfdb_dir = data_dir / "records500" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00456_hr").touch()

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

    # Create dummy waveform file path
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    for i in ids:
        (wfdb_dir / f"{i:05d}_lr").touch()

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

    # Create dummy waveform file path
    wfdb_dir = data_dir / "records100" / "00000"
    wfdb_dir.mkdir(parents=True)
    (wfdb_dir / "00789_lr").touch()

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
    with pytest.raises(ValueError, match="No valid records were loaded."):
        load_ptbxl_full(data_dir=str(data_dir), subsample_frac=1.0)


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
        # Empty strinigifed dictionary
        ("{}", "Unknown"),
    ],
)
def test_raw_to_five_class_cases(input_val, expected):
    result = raw_to_five_class(input_val)
    assert result == expected
