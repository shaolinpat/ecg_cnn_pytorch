import numpy as np
import pytest
from ecg_cnn.utils.validate import (
    validate_hparams_config,
    validate_hparams_formatting,
    validate_y_true_pred,
    validate_y_probs,
)


# ------------------------------------------------------------------------------
# def validate_hparams_config(
#     *,
#     model: str,
#     lr: float,
#     bs: int,
#     wd: float,
#     n_epochs: int,
#     n_folds: int,
# ) -> None:
# ------------------------------------------------------------------------------


def test_validate_hparams_config_all_valid_bare_minimum():
    # Should not raise
    validate_hparams_config(
        model="SomeModel", lr=0.001, bs=32, wd=0.01, n_epochs=10, n_folds=5
    )


def test_validate_hparams_config_model_empty():
    with pytest.raises(ValueError, match="Model must be a non-empty string."):
        validate_hparams_config(
            model="",
            lr="0.001",
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_model_not_string():
    with pytest.raises(ValueError, match="Model must be a non-empty string."):
        validate_hparams_config(
            model={},
            lr="0.001",
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_lr_not_int_or_float():
    with pytest.raises(ValueError, match="Learning rate must be positive int or float"):
        validate_hparams_config(
            model="SomeModel",
            lr="0.001",
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_lr_negative():
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        validate_hparams_config(
            model="SomeModel",
            lr=-0.001,
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_lr_too_small():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.0,
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_lr_too_large():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=10.001,
            bs=32,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_bs_not_int():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=32.0,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_bs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=0,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_bs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=4097,
            wd=0.0001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_wd_not_int_or_float():
    with pytest.raises(
        ValueError,
        match="Weight decay must be int or float",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd="0.0001",
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_wd_not_too_small():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=-0.1,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_wd_too_large():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=1.001,
            n_folds=2,
            n_epochs=3,
        )


def test_validate_hparams_config_n_folds_not_int():
    with pytest.raises(
        ValueError,
        match="n_folds must be a positive integer.",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds="2.0",
            n_epochs=3,
        )


def test_validate_hparams_config_n_folds_not_positive_int():
    with pytest.raises(
        ValueError,
        match="n_folds must be a positive integer.",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=0,
            n_epochs=3,
        )


def test_validate_hparams_config_n_folds_not_positive_int_but_float():
    with pytest.raises(
        ValueError,
        match="n_folds must be a positive integer.",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=3.0,
            n_epochs=3,
        )


def test_validate_hparams_config_n_epochs_not_int():
    with pytest.raises(
        ValueError,
        match=r"n_epochs must be int in range \[1, 1000\].",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=3,
            n_epochs=[3],
        )


def test_validate_hparams_config_n_epochs_not_int_but_float():
    with pytest.raises(
        ValueError,
        match=r"n_epochs must be int in range \[1, 1000\].",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=3,
            n_epochs=3.0,
        )


def test_validate_hparams_config_n_epochs_too_small():
    with pytest.raises(
        ValueError,
        match=r"n_epochs must be int in range \[1, 1000\].",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=3,
            n_epochs=0,
        )


def test_validate_hparams_config_n_epochs_too_large():
    with pytest.raises(
        ValueError,
        match=r"n_epochs must be int in range \[1, 1000\].",
    ):
        validate_hparams_config(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.4,
            n_folds=3,
            n_epochs=1001,
        )


# ------------------------------------------------------------------------------
# def validate_hparams_formatting(
#     *,
#     model: str,
#     lr: float,
#     bs: int,
#     wd: float,
#     prefix: str,
#     fname_metric: str = "",
#     epoch: int | None = None,
#     fold: int | None = None,
# ) -> None:
# ------------------------------------------------------------------------------


def test_validate_hparams_formatting_all_valid_bare_minimum():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel", lr=0.001, bs=32, wd=0.01, epoch=10, prefix="test"
    )


def test_validate_hparams_formatting_all_valid_minimal_plus_n_folds():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epoch=12,
        prefix="test",
        fold=1,
    )


def test_validate_hparams_formatting_all_valid_full():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epoch=100,
        prefix="test",
        fname_metric="loss",
        fold=1,
    )


def test_validate_hparams_formatting_all_valid_full_fname_metric_none():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epoch=100,
        prefix="test",
        fname_metric=None,
        fold=1,
    )


def test_validate_hparams_formatting_all_valid_full_fname_metric_spaces_only():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.01,
        epoch=100,
        prefix="test",
        fname_metric="   ",
        fold=44,
    )


def test_validate_hparams_formatting_all_valid_full_fold_validly_missing():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.0001,
        epoch=3,
        prefix="good",
    )


def test_validate_hparams_formatting_fold_None():
    # Should not raise
    validate_hparams_formatting(
        model="SomeModel",
        lr=0.001,
        bs=32,
        wd=0.0001,
        fold=None,
        epoch=3,
        prefix="bad",
    )


def test_validate_hparams_formatting_model_empty():
    with pytest.raises(ValueError, match="Model must be a non-empty string."):
        validate_hparams_formatting(
            model="",
            lr="0.001",
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_model_not_string():
    with pytest.raises(ValueError, match="Model must be a non-empty string."):
        validate_hparams_formatting(
            model=[1, 2],
            lr="0.001",
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_lr_not_int_or_float():
    with pytest.raises(ValueError, match="Learning rate must be positive int or float"):
        validate_hparams_formatting(
            model="SomeModel",
            lr="0.001",
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_lr_negative():
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        validate_hparams_formatting(
            model="SomeModel",
            lr=-0.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_lr_too_small():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.0,
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_lr_too_large():
    with pytest.raises(
        ValueError,
        match=r"Learning rate must be positive int or float in range \[1e-6, 1\.0\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=10.001,
            bs=32,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_bs_not_int():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=32.0,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_bs_too_small():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=0,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_bs_too_large():
    with pytest.raises(
        ValueError,
        match=r"Batch size must be an integer in range \[1, 4096\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=4097,
            wd=0.0001,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_wd_not_int_or_float():
    with pytest.raises(
        ValueError,
        match="Weight decay must be int or float",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd="0.0001",
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_wd_not_too_small():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=-0.1,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_wd_too_large():
    with pytest.raises(
        ValueError,
        match=r"Weight decay must be int or float in range \[0.0, 1.0\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=2,
            fold=2,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_fold_not_int_but_string():
    with pytest.raises(
        ValueError,
        match="fold must be a positive integer.",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold="2",
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_fold_not_int_but_float():
    with pytest.raises(
        ValueError,
        match="fold must be a positive integer.",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2.0,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_fold_too_small():
    with pytest.raises(
        ValueError,
        match="fold must be a positive integer.",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=0,
            epoch=3,
            prefix="bad",
        )


def test_validate_hparams_formatting_epochs_not_int_but_string():
    with pytest.raises(
        ValueError,
        match=r"epoch must be an integer in range \[1, 1000\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epoch="3",
            prefix="bad",
        )


def test_validate_hparams_formatting_epochs_not_int_but_float():
    with pytest.raises(
        ValueError,
        match=r"epoch must be an integer in range \[1, 1000\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epoch=3.0,
            prefix="bad",
        )


def test_validate_hparams_formatting_epochs_too_small():
    with pytest.raises(
        ValueError,
        match=r"epoch must be an integer in range \[1, 1000\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epoch=0,
            prefix="bad",
        )


def test_validate_hparams_formatting_epochs_too_large():
    with pytest.raises(
        ValueError,
        match=r"epoch must be an integer in range \[1, 1000\]",
    ):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=128,
            wd=0.01,
            fold=2,
            epoch=1001,
            prefix="bad",
        )


def test_validate_hparams_formatting_prefix_not_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        validate_hparams_formatting(
            model="SomeModel", lr=0.001, bs=32, wd=0.001, fold=1, epoch=10, prefix=3
        )


def test_validate_hparams_formatting_prefix_empty_string():
    with pytest.raises(ValueError, match="Prefix must be a non-empty string"):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=32,
            wd=0.001,
            fold=20,
            epoch=10,
            prefix="",
        )


def test_validate_hparams_formatting_fname_metric_not_string():
    with pytest.raises(ValueError, match="fname_metric must be a string"):
        validate_hparams_formatting(
            model="SomeModel",
            lr=0.001,
            bs=32,
            wd=0.001,
            fold=40,
            epoch=10,
            prefix="bad",
            fname_metric=3,
        )


# ------------------------------------------------------------------------------
# def validate_y_true_pred(y_true, y_pred):
# ------------------------------------------------------------------------------


def test_validate_y_true_pred_valid_lists():
    validate_y_true_pred([0, 1, 2], [1, 0, 2])  # Should pass


def test_validate_y_true_pred_valid_ndarrays():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([2, 1, 0])
    validate_y_true_pred(y_true, y_pred)  # Should pass


def test_validate_y_true_pred_mismatched_length():
    with pytest.raises(ValueError, match="same length"):
        validate_y_true_pred([0, 1], [0, 1, 2])


def test_validate_y_true_pred_invalid_type_y_true():
    with pytest.raises(ValueError, match="y_true must be list or ndarray"):
        validate_y_true_pred("not a list", [0, 1, 2])


def test_validate_y_true_pred_invalid_type_y_pred():
    with pytest.raises(ValueError, match="y_pred must be list or ndarray"):
        validate_y_true_pred([0, 1, 2], "not a list")


def test_validate_y_true_pred_non_integer_y_true():
    with pytest.raises(ValueError, match="y_true must contain only integer"):
        validate_y_true_pred([0, 1.5, 2], [0, 1, 2])


def test_validate_y_true_pred_non_integer_y_pred():
    with pytest.raises(ValueError, match="y_pred must contain only integer"):
        validate_y_true_pred([0, 1, 2], [0, 1, 2.2])


def test_validate_y_true_pred_bool_in_y_true():
    with pytest.raises(ValueError, match="y_true must contain only integer"):
        validate_y_true_pred([0, True, 2], [0, 1, 2])


def test_validate_y_true_pred_bool_in_y_pred():
    with pytest.raises(ValueError, match="y_pred must contain only integer"):
        validate_y_true_pred([0, 1, 2], [True, 1, 2])


# ------------------------------------------------------------------------------
# def validate_y_probs(y_probs):
# ------------------------------------------------------------------------------


def test_validate_y_probs_valid_list():
    validate_y_probs([0.0, 0.5, 1.0])  # Should pass


def test_validate_y_probs_valid_ndarray():
    validate_y_probs(np.array([0.25, 0.75]))  # Should pass


def test_validate_y_probs_invalid_type():
    with pytest.raises(ValueError, match="y_probs must be list or ndarray"):
        validate_y_probs("not a list")


def test_validate_y_probs_contains_int():
    with pytest.raises(ValueError, match="must contain only float"):
        validate_y_probs([0.0, 1, 0.5])  # int is not a float


def test_validate_y_probs_contains_bool():
    with pytest.raises(ValueError, match="must contain only float"):
        validate_y_probs([0.0, True, 0.5])  # bool is subclass of int


def test_validate_y_probs_out_of_range_low():
    with pytest.raises(ValueError, match=r"probabilities in \[0.0, 1.0\]"):
        validate_y_probs([-0.1, 0.5, 0.9])


def test_validate_y_probs_out_of_range_high():
    with pytest.raises(ValueError, match=r"probabilities in \[0.0, 1.0\]"):
        validate_y_probs([0.0, 1.1, 0.5])


def test_validate_y_probs_empty_list():
    validate_y_probs([])  # Should pass — empty is still valid


def test_validate_y_probs_ndarray_with_integers():
    arr = np.array([0, 1, 1], dtype=int)
    with pytest.raises(ValueError, match="must contain only float"):
        validate_y_probs(arr)


def test_validate_y_probs_ndarray_with_booleans():
    arr = np.array([True, False, True], dtype=bool)
    with pytest.raises(ValueError, match="must contain only float"):
        validate_y_probs(arr)


def test_validate_y_probs_ndarray_too_many_dimensions():
    arr = np.ones((2, 2, 2), dtype=float)
    with pytest.raises(ValueError, match="must be 1D or 2D"):
        validate_y_probs(arr)
