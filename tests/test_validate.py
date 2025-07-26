import pytest
from ecg_cnn.utils.validate import validate_hparams

# ------------------------------------------------------------------------------
# def validate_hparams(
#     lr: float,
#     bs: int,
#     wd: float,
#     fold: int,
#     epochs: int,
#     prefix: str,
#     fname_metric: str = "",
# ) -> None:
# ------------------------------------------------------------------------------


def test_validate_hparams_all_valid_bare_minimum():
    # Should not raise
    validate_hparams(
        model="SomeModel", lr=0.001, bs=32, wd=0.01, epochs=10, prefix="test"
    )


def test_validate_hparams_all_valid_minimal_plus_fold():
    # Should not raise
    validate_hparams(
        model="SomeModel", lr=0.001, bs=32, wd=0.01, epochs=10, prefix="test", fold=1
    )


def test_validate_hparams_all_valid_full():
    # Should not raise
    validate_hparams(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epochs=100,
        prefix="test",
        fname_metric="loss",
        fold=0,
    )


def test_validate_hparams_all_valid_full_fname_metric_none():
    # Should not raise
    validate_hparams(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epochs=100,
        prefix="test",
        fname_metric=None,
        fold=1,
    )


def test_validate_hparams_all_valid_full_fname_metric_spaces_only():
    # Should not raise
    validate_hparams(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epochs=100,
        prefix="test",
        fname_metric="   ",
        fold=44,
    )


def test_validate_hparams_all_valid_full_fold_none():
    # Should not raise
    validate_hparams(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epochs=1000,
        prefix="test",
        fname_metric="loss",
        fold=None,
    )


def test_validate_hparams_all_valid_full_fold_validly_missing():
    # Should not raise
    validate_hparams(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.0001,
        epochs=3,
        prefix="good",
    )


def test_validate_hparams_lr_not_int_or_float():
    with pytest.raises(ValueError, match="Learning rate must be positive int or float"):
        validate_hparams(
            model="SomeModel",
            lr="0.001",
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_negative():
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        validate_hparams(
            model="SomeModel",
            lr=-0.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_too_small():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.0,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_lr_too_large():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=10.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_not_int():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=32.0,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=0,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_bs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=4097,
            wd=0.0001,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_int_or_float():
    with pytest.raises(
        ValueError,
        match="Weight decay must be int or float",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd="0.0001",
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_too_small():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=-0.1,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_wd_not_too_large():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=2,
            fold=2,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_not_int_but_string():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold="2",
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_not_int_but_float():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2.0,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_fold_too_small():
    with pytest.raises(
        ValueError,
        match="Fold number must be a non-negative integer",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=-1,
            epochs=3,
            prefix="bad",
        )


def test_validate_hparams_epochs_not_int_but_string():
    with pytest.raises(
        ValueError,
        match="Epochs must be an integer",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs="3",
            prefix="bad",
        )


def test_validate_hparams_epochs_not_int_but_float():
    with pytest.raises(
        ValueError,
        match="Epochs must be an integer",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=3.0,
            prefix="bad",
        )


def test_validate_hparams_epochs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Epochs must be an integer in range \[1, 1000\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=0,
            prefix="bad",
        )


def test_validate_hparams_epochs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Epochs must be an integer in range \[1, 1000\]",
    ):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epochs=1001,
            prefix="bad",
        )


def test_validate_hparams_prefix_not_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        validate_hparams(
            model="SomeModel", lr=0.001, bs=32, wd=0.001, fold=0, epochs=10, prefix=3
        )


def test_validate_hparams_prefix_empty_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        validate_hparams(
            model="SomeModel", lr=0.001, bs=32, wd=0.001, fold=0, epochs=10, prefix=""
        )


def test_validate_hparams_fname_metric_not_string():
    with pytest.raises(ValueError, match="Metric name must be a string"):
        validate_hparams(
            model="SomeModel",
            lr=0.001,
            bs=32,
            wd=0.001,
            fold=0,
            epochs=10,
            prefix="bad",
            fname_metric=3,
        )
