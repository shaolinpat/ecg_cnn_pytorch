# ------------------------------------------------------------------------------
# Shared validation for common hyperparameters
# ------------------------------------------------------------------------------
def validate_hparams(
    lr: float,
    bs: int,
    wd: float,
    fold: int,
    epochs: int,
    prefix: str,
    fname_metric: str = "",
) -> None:
    """
    Validates hyperparameters used for filename formatting or training configuration.

    Parameters
    ----------
    lr : float
        Learning rate. Must be in range [1e-6, 1.0].
    bs : int
        Batch size. Must be in range [1, 4096].
    wd : float
        Weight decay. Must be in range [0.0, 1.0].
    fold : int
        Fold number in cross-validation. Must be non-negative.
    epochs : int
        Number of training epochs. Must be in range [1, 1000].
    prefix : str
        Prefix string indicating model phase or purpose (e.g., "final", "best").
    fname_metric : str, optional
        Optional metric name (e.g., "loss", "accuracy") to include in filename.

    Returns
    -------
    Nothing is returned.

    Raises
    ------
    ValueError
        If any of the provided values are of incorrect type or out of valid range.
    """
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
    if not isinstance(fold, int) or fold < 0:
        raise ValueError(f"Fold number must be a non-negative integer. Got: {fold}")
    if not isinstance(epochs, int) or not (1 <= epochs <= 1000):
        raise ValueError(f"Epochs must be an integer in range [1, 1000]. Got: {epochs}")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError(f"Prefix must be a non-empty string. Got: {prefix!r}")
    if fname_metric is not None and not isinstance(fname_metric, str):
        raise ValueError("Metric name must be a string")
