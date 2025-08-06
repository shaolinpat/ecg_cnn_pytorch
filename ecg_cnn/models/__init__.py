from .model_utils import ECGConvNet, ECGResNet, ECGInceptionNet

MODEL_CLASSES = {
    "ECGConvNet": ECGConvNet,
    "ECGResNet": ECGResNet,
    "ECGInceptionNet": ECGInceptionNet,
}
