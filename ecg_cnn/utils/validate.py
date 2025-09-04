# utils/validate.py

import numpy as np


INT_TYPES = (int, np.integer)
FLOAT_TYPES = (float, np.floating)


def validate_hparams_config(
    *,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    n_epochs: int,
    n_folds: int,
) -> None:
    """
    Validates hyperparameters used for training configuration.

    Parameters
    ----------
    model : str
        Model string used for identification and plotting (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    n_epochs : int
        Total number of training epochs. Must be in range [1, 1000].
    n_folds : int
        Total number of cross-validation folds. Must be a positive integer.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any parameter is of incorrect type or out of range.

    Note
    ----
    Use `validate_hparams_formatting()` for plot or filename validation.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Model must be a non-empty string. Got: {model!r}")
    if not isinstance(lr, (float, int)) or not (1e-6 <= lr <= 1.0):
        raise ValueError(
            f"Learning rate must be positive int or float in range [1e-6, 1.0]. Got: {lr}"
        )
    if not isinstance(bs, int) or not (1 <= bs <= 4096):
        raise ValueError(f"Batch size must be an integer in range [1, 4096]. Got: {bs}")
    if not isinstance(wd, (float, int)) or not (0.0 <= wd <= 1.0):
        raise ValueError(
            f"Weight decay must be int or float in range [0.0, 1.0]. Got: {wd}"
        )
    if not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError(f"n_folds must be a positive integer. Got: {n_folds}")
    if not isinstance(n_epochs, int) or not (1 <= n_epochs <= 1000):
        raise ValueError(f"n_epochs must be int in range [1, 1000]. Got: {n_epochs}")


def validate_hparams_formatting(
    *,
    model: str,
    lr: float,
    bs: int,
    wd: float,
    prefix: str,
    fname_metric: str = "",
    epoch: int | None = None,
    fold: int | None = None,
) -> None:
    """
    Validates hyperparameters used for plot titles and filenames.

    Parameters
    ----------
    model : str
        Model string used for identification and plotting (e.g., "ECGConvNet").
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    prefix : str
        String prefix for context (e.g., "final", "best"). Must be non-empty.
    fname_metric : str, optional
        Metric name for filenames (e.g., "loss", "accuracy", "f1").
    epoch : int or None, optional
        Epoch that produced the result. Must be in range [1, 1000] if provided.
    fold : int or None, optional
        Fold that generated the result. Must be a positive integer if provided.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any parameter is of incorrect type or out of range.

    Note
    ----
    Use `validate_hparams_config()` for full training configuration validation.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"Model must be a non-empty string. Got: {model!r}")
    if not isinstance(lr, (float, int)) or not (1e-6 <= lr <= 1.0):
        raise ValueError(
            f"Learning rate must be positive int or float in range [1e-6, 1.0]. Got: {lr}"
        )
    if not isinstance(bs, int) or not (1 <= bs <= 4096):
        raise ValueError(f"Batch size must be an integer in range [1, 4096]. Got: {bs}")
    if not isinstance(wd, (float, int)) or not (0.0 <= wd <= 1.0):
        raise ValueError(
            f"Weight decay must be int or float in range [0.0, 1.0]. Got: {wd}"
        )
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string. Got: {prefix!r}")
    if fname_metric is not None and not isinstance(fname_metric, str):
        raise ValueError("fname_metric must be a string.")
    if epoch is not None:
        if not isinstance(epoch, int) or not (1 <= epoch <= 1000):
            raise ValueError(
                f"epoch must be an integer in range [1, 1000]. Got: {epoch}"
            )
    if fold is not None:
        if not isinstance(fold, int) or fold < 1:
            raise ValueError(f"fold must be a positive integer. Got: {fold}")


def validate_y_true_pred(y_true, y_pred):
    """
    Validate that y_true and y_pred are sequences (list or ndarray) of equal length,
    and that all their elements are non-boolean integers.

    Parameters
    ----------
    y_true : list[int] or np.ndarray
        Ground-truth labels. Must be a list or NumPy array of integers (not bools).
    y_pred : list[int] or np.ndarray
        Predicted labels. Must be a list or NumPy array of integers (not bools).

    Raises
    ------
    ValueError
        If either y_true or y_pred is not a list or ndarray.
    ValueError
        If y_true and y_pred do not have the same length.
    ValueError
        If any element in y_true or y_pred is not a non-boolean integer.

    Return
    ------
    None
        This function performs validation only and returns nothing.
    """
    if not isinstance(y_true, (list, np.ndarray)):
        raise ValueError(f"y_true must be list or ndarray, got {type(y_true)}")
    if not isinstance(y_pred, (list, np.ndarray)):
        raise ValueError(f"y_pred must be list or ndarray, got {type(y_pred)}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}."
        )
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_true):
        raise ValueError("y_true must contain only integer values (not bools)")
    if not all(isinstance(i, INT_TYPES) and not isinstance(i, bool) for i in y_pred):
        raise ValueError("y_pred must contain only integer values (not bools)")


def validate_y_probs(y_probs):
    """
    Validate that y_probs is a list or ndarray of floats within the range [0.0, 1.0].

    Parameters
    ----------
    y_probs : list[float] or np.ndarray
        A list or array of predicted probabilities. Each value must be a float
        between 0.0 and 1.0 inclusive.

    Raises
    ------
    ValueError
        If y_probs is not a list or NumPy array.
    ValueError
        If any element is not a float.
    ValueError
        If any value is outside the range [0.0, 1.0].

    Returns
    -------
    None
        This function performs validation only and returns nothing.
    """
    if not isinstance(y_probs, (list, np.ndarray)):
        raise ValueError(f"y_probs must be list or ndarray, got {type(y_probs)}")

    if isinstance(y_probs, list):
        if not all(isinstance(p, float) for p in y_probs):
            raise ValueError("y_probs must contain only float values")
        arr = np.asarray(y_probs, dtype=float)
    else:  # ndarray
        arr = np.asarray(y_probs)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("y_probs must contain only float values")

    if arr.ndim > 2:
        raise ValueError("y_probs must be 1D or 2D")

    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("y_probs must contain probabilities in [0.0, 1.0]")
