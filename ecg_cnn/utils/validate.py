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
    if not isinstance(n_epochs, int) or not (1 <= n_epochs <= 1000):
        raise ValueError(f"n_epochs must be in [1, 1000]. Got: {n_epochs}")
    if not isinstance(n_folds, int) or n_folds < 1:
        raise ValueError(f"n_folds must be a positive integer. Got: {n_folds}")


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
