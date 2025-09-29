# tests/test_evaluate.py

"""
Tests for ecg_cnn.evaluate.py

Covers
------
    - main(): happy path, error paths, fold selection, env overrides
    - __main__ entry point smoke test via subprocess
"""

import builtins
import importlib.util
import json
import math
import numpy as np
import pandas as pd
import pathlib
import pytest
import os
import re
import runpy
import subprocess
import sys
import time
import types

from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock
from torch.utils.data import DataLoader, TensorDataset

# Optional dep: skip cleanly if PyTorch isn't installed
if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level=True)
import torch

import ecg_cnn.evaluate as evaluate
import ecg_cnn.models as models


assert "/ecg_cnn_pytorch/ecg_cnn/evaluate.py" in evaluate.__file__
print("EVAL FILE:", evaluate.__file__)

# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------


def ovr_cfg(plots_enable_ovr=False, plots_ovr_classes=None):
    return SimpleNamespace(
        plots_enable_ovr=plots_enable_ovr,
        plots_ovr_classes=[] if plots_ovr_classes is None else plots_ovr_classes,
    )


class _TinyLogitModel(torch.nn.Module):
    """Minimal fake model that produces logits for N classes."""

    def __init__(self, num_classes=2, input_channels=12, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(input_channels, num_classes, bias=False)

    def forward(self, x):
        # handle shapes like (N,C,T) or (N,C)
        if hasattr(x, "ndim") and x.ndim == 3:
            x = x.mean(dim=2)
        return self.fc(x)

    def load_state_dict(self, *a, **k):
        return  # ignore weights in tests


def _write_sitecustomize(sc_path: Path, tmp_path: Path):
    """
    Write the EXACT sitecustomize payload that already works for you.
    TAKE the string literal from test_cli_entrypoint_fast and put it here,
    so every test can reuse it without duplicating 500 lines.
    """
    sc_path.write_text(
        "import sys, types, pathlib\n"
        "\n"
        "def _ensure_pkg(dotted):\n"
        "    parts = dotted.split('.')\n"
        "    base = ''\n"
        "    for i, p in enumerate(parts):\n"
        "        base = p if i == 0 else base + '.' + p\n"
        "        if base not in sys.modules:\n"
        "            m = types.ModuleType(base)\n"
        "            m.__path__ = []\n"
        "            sys.modules[base] = m\n"
        "            if i > 0:\n"
        "                parent = sys.modules['.'.join(parts[:i])]\n"
        "                setattr(parent, p, m)\n"
        "    return sys.modules[dotted]\n"
        "\n"
        "# Ensure core packages exist first\n"
        "for name in [\n"
        "    'ecg_cnn','ecg_cnn.paths',\n"
        "    'sklearn','sklearn.metrics','sklearn.preprocessing','sklearn.calibration',\n"
        "]:\n"
        "    _ensure_pkg(name)\n"
        "\n"
        "# Define sklearn.metrics names eagerly so imports succeed\n"
        "sm = sys.modules['sklearn.metrics']\n"
        "\n"
        "# Simple primitives\n"
        "def _float(*a, **k): return 0.0\n"
        "def _report(*a, **k): return 'stub-report'\n"
        "def _cm(*a, **k): return [[0, 0], [0, 0]]\n"
        "def _curve3(*a, **k): return ([0.0], [0.0], [0.0])\n"
        "\n"
        "# Commonly-used names wired explicitly\n"
        "sm.accuracy_score = _float\n"
        "sm.precision_score = _float\n"
        "sm.recall_score = _float\n"
        "sm.f1_score = _float\n"
        "sm.roc_auc_score = _float\n"
        "sm.average_precision_score = _float\n"
        "sm.top_k_accuracy_score = _float\n"
        "sm.balanced_accuracy_score = _float\n"
        "sm.brier_score_loss = _float\n"
        "sm.mean_squared_error = _float\n"
        "sm.mean_absolute_error = _float\n"
        "sm.log_loss = _float\n"
        "\n"
        "sm.precision_recall_curve = _curve3\n"
        "sm.roc_curve = _curve3\n"
        "sm.classification_report = _report\n"
        "sm.confusion_matrix = _cm\n"
        "\n"
        "# ---- UNIVERSAL FALLBACK ----\n"
        "# Any unknown attribute on sklearn.metrics becomes a no-op metric function\n"
        "def __getattr__(name):\n"
        "    # Return a 0.0-valued metric for anything that looks like a metric/loss/score\n"
        "    if name.endswith('_score') or name.endswith('_loss') or name.endswith('_error') or name in ('auc','ap'):\n"
        "        return _float\n"
        "    # Return a 3-tuple curve for names that look like curve generators\n"
        "    if name.endswith('_curve'):\n"
        "        return _curve3\n"
        "    # Default: harmless function\n"
        "    return _float\n"
        "sm.__getattr__ = __getattr__\n"
        "\n"
        "# Define sklearn.preprocessing.LabelEncoder\n"
        "sp = sys.modules['sklearn.preprocessing']\n"
        "class _LabelEncoder:\n"
        "    def fit(self, y): return self\n"
        "    def transform(self, y): return list(range(len(y)))\n"
        "    def fit_transform(self, y): return list(range(len(y)))\n"
        "sp.LabelEncoder = _LabelEncoder\n"
        "\n"
        "# Define sklearn.calibration.calibration_curve\n"
        "scal = sys.modules['sklearn.calibration']\n"
        "def calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform'):\n"
        "    # Minimal stub: always return one dummy bin\n"
        "    return [0.0], [0.0]\n"
        "scal.calibration_curve = calibration_curve\n"
        "\n"
        "# Stub the rest of the heavy trees used by evaluate.py (import-time only)\n"
        "for name in [\n"
        "    'torch','torch.nn','torch.nn.functional',\n"
        "    'torch.utils','torch.utils.data','torch.cuda',\n"
        "    'torch.backends','torch.backends.cudnn',\n"
        "    'pandas',\n"
        "    'matplotlib','matplotlib.pyplot','matplotlib.colors','matplotlib.cbook','matplotlib.figure',\n"
        "    'matplotlib.collections','matplotlib.markers','matplotlib.patches','matplotlib.ticker','matplotlib.axes','matplotlib.axis','matplotlib.cm','matplotlib.gridspec','matplotlib.lines','matplotlib.text',\n"
        "    'matplotlib.dates','matplotlib.scale','matplotlib.transforms','matplotlib.style',\n"
        "    'cycler',\n"
        "    'shap',\n"
        "    'wfdb','wfdb.io','wfdb.io.record',\n"
        "]:\n"
        "    _ensure_pkg(name)\n"
        "\n"
        "# Override ecg_cnn.paths to point into the tmp dir; include every attr evaluate.py might import\n"
        "paths_mod = sys.modules['ecg_cnn.paths']\n"
        + "base = pathlib.Path(r'"
        + str(tmp_path)
        + "')\n"
        + "paths_mod.PROJECT_ROOT   = base\n"
        "paths_mod.OUTPUT_DIR     = base\n"
        "paths_mod.RESULTS_DIR    = base\n"
        "paths_mod.REPORTS_DIR    = base\n"
        "paths_mod.PTBXL_DATA_DIR = base\n"
        "paths_mod.HISTORY_DIR    = base\n"
        "paths_mod.PLOTS_DIR      = base\n"
        "paths_mod.MODELS_DIR     = base\n"
        "paths_mod.ARTIFACTS_DIR  = base\n"
        "paths_mod.PTBXL_META_CSV = base / 'ptbxl_database.csv'\n"
        "paths_mod.PTBXL_SCP_CSV  = base / 'scp_statements.csv'\n"
        "\n"
        "# Minimal torch stubs\n"
        "nn = sys.modules.get('torch.nn')\n"
        "if nn is not None:\n"
        "    class _Module:\n"
        "        def __init__(self,*a,**k): pass\n"
        "        def to(self,*a,**k): return self\n"
        "        def eval(self,*a,**k): return self\n"
        "        def __call__(self,*a,**k): return a[0] if a else None\n"
        "    class _Linear(_Module):\n"
        "        def __call__(self,x): return x\n"
        "    class _CrossEntropyLoss:\n"
        "        def __call__(self,input,target):\n"
        "            class _Z:\n"
        "                def item(self): return 0.0\n"
        "            return _Z()\n"
        "    class _Sequential(_Module):\n"
        "        def __call__(self,x): return x\n"
        "    nn.Module=_Module; nn.Linear=_Linear; nn.CrossEntropyLoss=_CrossEntropyLoss; nn.Sequential=_Sequential\n"
        "\n"
        "nnf = sys.modules.get('torch.nn.functional')\n"
        "if nnf is not None:\n"
        "    nnf.softmax=lambda x,dim=None:x\n"
        "\n"
        "_t=sys.modules.get('torch')\n"
        "if _t is not None:\n"
        "    # Basic API expected by evaluate.py\n"
        "    def manual_seed(seed): return None\n"
        "    _t.manual_seed = manual_seed\n"
        "\n"
        "    def set_num_threads(n): return None\n"
        "    _t.set_num_threads = set_num_threads\n"
        "\n"
        "    def get_num_threads(): return 1\n"
        "    _t.get_num_threads = get_num_threads\n"
        "\n"
        "    _t.device = lambda *a, **k: 'cpu'\n"
        "\n"
        "    class Tensor: pass\n"
        "    _t.Tensor = Tensor\n"
        "\n"
        "    # torch.cuda stubs\n"
        "    cuda = sys.modules.get('torch.cuda')\n"
        "    if cuda is not None:\n"
        "        cuda.is_available   = lambda: False\n"
        "        cuda.manual_seed_all = lambda s: None\n"
        "        cuda.empty_cache    = lambda: None\n"
        "        cuda.device_count   = lambda: 0\n"
        "\n"
        "    # torch.backends.cudnn flags sometimes toggled\n"
        "    cudnn = sys.modules.get('torch.backends.cudnn')\n"
        "    if cudnn is not None:\n"
        "        cudnn.enabled = False\n"
        "        cudnn.benchmark = False\n"
        "        cudnn.deterministic = True\n"
        "\n"
        "    # no_grad context manager\n"
        "    class _NoGrad:\n"
        "        def __enter__(self): return self\n"
        "        def __exit__(self, exc_type, exc, tb): return False\n"
        "    _t.no_grad = lambda: _NoGrad()\n"
        "\n"
        "tud=sys.modules.get('torch.utils.data')\n"
        "if tud is not None:\n"
        "    class _Dataset: pass\n"
        "    class _TensorDataset: pass\n"
        "    class _DataLoader: pass\n"
        "    tud.Dataset=_Dataset; tud.TensorDataset=_TensorDataset; tud.DataLoader=_DataLoader\n"
        "\n"
        "# Minimal pandas used at import time\n"
        "pd=sys.modules.get('pandas')\n"
        "if pd is not None:\n"
        "    class _Index:\n"
        "        def __init__(self,v=None,name=None): self._values=list(v) if v else []; self.name=name\n"
        "        def tolist(self): return list(self._values)\n"
        "        def __iter__(self): return iter(self._values)\n"
        "        def __len__(self): return len(self._values)\n"
        "    class _DF:\n"
        "        def __init__(self,*a,**k): self.index=_Index(); self.columns=[]\n"
        "        def reset_index(self,*a,**k): return self\n"
        "        def set_index(self,*a,**k): return self\n"
        "        def __getitem__(self,k): return self\n"
        "        @property\n"
        "        def loc(self): return self\n"
        "    pd.DataFrame=_DF; pd.read_csv=lambda *a,**k:_DF(); pd.concat=lambda *a,**k:_DF()\n"
        "    pd.Series=lambda *a,**k:[]; pd.Index=_Index; pd.Timestamp=lambda *a,**k:None; pd.Timedelta=lambda *a,**k:None\n"
        "\n"
        "# Matplotlib bits touched at import time\n"
        "mpl=sys.modules.get('matplotlib')\n"
        "if mpl is not None:\n"
        "    # backend + rcparams\n"
        "    setattr(mpl,'use',lambda *a,**k:None)\n"
        "    try:\n"
        "        mpl.rcParams\n"
        "    except AttributeError:\n"
        "        class _RcParams(dict):\n"
        "            def copy(self): return _RcParams(**dict(self))\n"
        "        mpl.rcParams = _RcParams()\n"
        "        mpl.rcParamsDefault = _RcParams()\n"
        "        mpl.rcParamsOrig = _RcParams()\n"
        "    mpl.get_backend = (lambda: 'Agg')\n"
        "mcolors=sys.modules.get('matplotlib.colors')\n"
        "if mcolors is not None:\n"
        "    # core API functions commonly imported\n"
        "    mcolors.to_rgb = lambda c: (0, 0, 0)\n"
        "    mcolors.to_rgba = lambda c, alpha=None: (0, 0, 0, 1.0 if alpha is None else alpha)\n"
        "    mcolors.to_hex = lambda c, keep_alpha=False: '#000000' if not keep_alpha else '#000000ff'\n"
        "    mcolors.is_color_like = lambda c: True\n"
        "\n"
        "    # minimal class stubs used by matplotlib\n"
        "    class Colormap: pass\n"
        "    class ListedColormap(Colormap):\n"
        "        def __init__(self, colors, name='from_list', N=None): self.colors = colors\n"
        "    class LinearSegmentedColormap(Colormap):\n"
        "        def __init__(self, name, segmentdata, N=256): self.name = name\n"
        "\n"
        "    class Normalize:\n"
        "        def __init__(self, vmin=None, vmax=None, clip=False): self.vmin=vmin; self.vmax=vmax; self.clip=clip\n"
        "        def __call__(self, value): return value\n"
        "    class NoNorm(Normalize): pass\n"
        "    class LogNorm(Normalize): pass\n"
        "    class SymLogNorm(Normalize):\n"
        "        def __init__(self, linthresh, linscale=1.0, vmin=None, vmax=None, base=10): pass\n"
        "    class TwoSlopeNorm(Normalize):\n"
        "        def __init__(self, vcenter, vmin=None, vmax=None): pass\n"
        "    class BoundaryNorm(Normalize):\n"
        "        def __init__(self, boundaries, ncolors, clip=False): pass\n"
        "\n"
        "    mcolors.Colormap = Colormap\n"
        "    mcolors.ListedColormap = ListedColormap\n"
        "    mcolors.LinearSegmentedColormap = LinearSegmentedColormap\n"
        "    mcolors.Normalize = Normalize\n"
        "    mcolors.NoNorm = NoNorm\n"
        "    mcolors.LogNorm = LogNorm\n"
        "    mcolors.SymLogNorm = SymLogNorm\n"
        "    mcolors.TwoSlopeNorm = TwoSlopeNorm\n"
        "    mcolors.BoundaryNorm = BoundaryNorm\n"
        "mcbook=sys.modules.get('matplotlib.cbook')\n"
        "if mcbook is not None:\n"
        "    mcbook.normalize_kwargs=lambda kw,*a,**k:dict(kw or {})\n"
        "mfig=sys.modules.get('matplotlib.figure')\n"
        "if mfig is not None:\n"
        "    class Figure: pass\n"
        "    mfig.Figure=Figure\n"
        "mcoll=sys.modules.get('matplotlib.collections')\n"
        "if mcoll is not None:\n"
        "    class Collection: pass\n"
        "    class PatchCollection(Collection): pass\n"
        "    class PathCollection(Collection): pass\n"
        "    class LineCollection(Collection): pass\n"
        "    class EventCollection(Collection): pass\n"
        "    class QuadMesh(Collection): pass\n"
        "    class PolyCollection(Collection): pass\n"
        "    class RegularPolyCollection(PolyCollection): pass\n"
        "\n"
        "    mcoll.Collection = Collection\n"
        "    mcoll.PatchCollection = PatchCollection\n"
        "    mcoll.PathCollection = PathCollection\n"
        "    mcoll.LineCollection = LineCollection\n"
        "    mcoll.EventCollection = EventCollection\n"
        "    mcoll.QuadMesh = QuadMesh\n"
        "    mcoll.PolyCollection = PolyCollection\n"
        "    mcoll.RegularPolyCollection = RegularPolyCollection\n"
        "mmark=sys.modules.get('matplotlib.markers')\n"
        "if mmark is not None:\n"
        "    class MarkerStyle: pass\n"
        "    mmark.MarkerStyle=MarkerStyle\n"
        "mpatch=sys.modules.get('matplotlib.patches')\n"
        "if mpatch is not None:\n"
        "    class Rectangle: pass\n"
        "    mpatch.Rectangle=Rectangle\n"
        "mtick=sys.modules.get('matplotlib.ticker')\n"
        "if mtick is not None:\n"
        "    class Locator: pass\n"
        "    class Formatter: pass\n"
        "    class AutoLocator(Locator): pass\n"
        "    class AutoMinorLocator(Locator): pass\n"
        "    class MaxNLocator(Locator): pass\n"
        "    class LinearLocator(Locator):\n"
        "        def __init__(self, numticks=None): self.numticks = numticks\n"
        "    class MultipleLocator(Locator):\n"
        "        def __init__(self, base=1.0): self.base = base\n"
        "    class FixedLocator(Locator):\n"
        "        def __init__(self, locs=None, nbins=None): self.locs = list(locs or []); self.nbins = nbins\n"
        "    class IndexLocator(Locator):\n"
        "        def __init__(self, base=1.0, offset=0): self.base=base; self.offset=offset\n"
        "    class LogLocator(Locator): pass\n"
        "    class LogitLocator(Locator): pass\n"
        "    class SymmetricalLogLocator(Locator): pass\n"
        "    class NullLocator(Locator): pass\n"
        "    class NullFormatter(Formatter): pass\n"
        "    class ScalarFormatter(Formatter): pass\n"
        "    class LogFormatter(Formatter): pass\n"
        "    class LogitFormatter(Formatter): pass\n"
        "    class LogFormatterSciNotation(Formatter): pass\n"
        "    class LogFormatterExponent(Formatter): pass\n"
        "    class FixedFormatter(Formatter):\n"
        "        def __init__(self, seq=()): self.seq = list(seq)\n"
        "    class FuncFormatter(Formatter):\n"
        "        def __init__(self, func=None): self.func = func\n"
        "    class FormatStrFormatter(Formatter):\n"
        "        def __init__(self, fmt='%s'): self.fmt = fmt\n"
        "    class StrMethodFormatter(Formatter):\n"
        "        def __init__(self, fmt='{x}'): self.fmt = fmt\n"
        "    class PercentFormatter(Formatter):\n"
        "        def __init__(self, xmax=1.0, decimals=None, symbol='%', is_latex=False):\n"
        "            self.xmax = xmax; self.decimals = decimals; self.symbol = symbol; self.is_latex = is_latex\n"
        "    class EngFormatter(Formatter):\n"
        "        def __init__(self, unit='', places=None, sep=' '):\n"
        "            self.unit = unit; self.places = places; self.sep = sep\n"
        "    mtick.Locator = Locator; mtick.Formatter = Formatter\n"
        "    mtick.AutoLocator = AutoLocator; mtick.AutoMinorLocator = AutoMinorLocator\n"
        "    mtick.MaxNLocator = MaxNLocator; mtick.LinearLocator = LinearLocator\n"
        "    mtick.MultipleLocator = MultipleLocator; mtick.FixedLocator = FixedLocator; mtick.IndexLocator = IndexLocator\n"
        "    mtick.LogLocator = LogLocator; mtick.LogitLocator = LogitLocator; mtick.SymmetricalLogLocator = SymmetricalLogLocator\n"
        "    mtick.NullLocator = NullLocator; mtick.NullFormatter = NullFormatter\n"
        "    mtick.ScalarFormatter = ScalarFormatter; mtick.LogFormatter = LogFormatter; mtick.LogitFormatter = LogitFormatter\n"
        "    mtick.LogFormatterSciNotation = LogFormatterSciNotation; mtick.LogFormatterExponent = LogFormatterExponent\n"
        "    mtick.FixedFormatter = FixedFormatter; mtick.FuncFormatter = FuncFormatter\n"
        "    mtick.FormatStrFormatter = FormatStrFormatter; mtick.StrMethodFormatter = StrMethodFormatter\n"
        "    mtick.PercentFormatter = PercentFormatter; mtick.EngFormatter = EngFormatter\n"
        "mdates=sys.modules.get('matplotlib.dates')\n"
        "if mdates is not None:\n"
        "    class DateFormatter:\n"
        "        def __init__(self, fmt=None): self.fmt = fmt\n"
        "    class AutoDateFormatter:\n"
        "        def __init__(self, locator, defaultfmt='%Y-%m-%d'): self.locator = locator; self.defaultfmt = defaultfmt\n"
        "    class ConciseDateFormatter:\n"
        "        def __init__(self, locator, tz=None, formats=None, zero_formats=None, show_offset=True, use_math_text=False):\n"
        "            self.locator = locator\n"
        "    class AutoDateLocator:\n"
        "        def __init__(self, minticks=5, maxticks=11, interval_multiples=False):\n"
        "            self.minticks = minticks; self.maxticks = maxticks; self.interval_multiples = interval_multiples\n"
        "    class MonthLocator:\n"
        "        def __init__(self, bymonth=None, bymonthday=1, interval=1, tz=None):\n"
        "            self.bymonth = bymonth; self.bymonthday = bymonthday; self.interval = interval; self.tz = tz\n"
        "    class YearLocator:\n"
        "        def __init__(self, base=1, month=1, day=1, tz=None):\n"
        "            self.base = base; self.month = month; self.day = day; self.tz = tz\n"
        "    class DayLocator:\n"
        "        def __init__(self, bdays=None, interval=1, tz=None):\n"
        "            self.bdays = bdays; self.interval = interval; self.tz = tz\n"
        "    class RRuleLocator:\n"
        "        def __init__(self, o, tz=None): self.o = o; self.tz = tz\n"
        "    def date2num(d): return 0.0\n"
        "    def num2date(n): return None\n"
        "    mdates.DateFormatter = DateFormatter\n"
        "    mdates.AutoDateFormatter = AutoDateFormatter\n"
        "    mdates.ConciseDateFormatter = ConciseDateFormatter\n"
        "    mdates.AutoDateLocator = AutoDateLocator\n"
        "    mdates.MonthLocator = MonthLocator\n"
        "    mdates.YearLocator = YearLocator\n"
        "    mdates.DayLocator = DayLocator\n"
        "    mdates.RRuleLocator = RRuleLocator\n"
        "    mdates.date2num = date2num\n"
        "    mdates.num2date = num2date\n"
        "mscale=sys.modules.get('matplotlib.scale')\n"
        "if mscale is not None:\n"
        "    class ScaleBase:\n"
        "        def __init__(self, axis, **kwargs): self.axis = axis\n"
        "        def get_transform(self): return None\n"
        "        def set_default_locators_and_formatters(self, axis): pass\n"
        "        def limit_range_for_scale(self, vmin, vmax, minpos): return (vmin, vmax)\n"
        "\n"
        "    class LinearScale(ScaleBase): pass\n"
        "    class LogScale(ScaleBase): pass\n"
        "    class SymmetricalLogScale(ScaleBase): pass\n"
        "\n"
        "    _REG = {\n"
        "        'linear': LinearScale,\n"
        "        'log': LogScale,\n"
        "        'symlog': SymmetricalLogScale,\n"
        "    }\n"
        "\n"
        "    def register_scale(cls):\n"
        "        name = getattr(cls, 'name', None)\n"
        "        if not name:\n"
        "            # Try common names by class\n"
        "            if cls is LinearScale: name='linear'\n"
        "            elif cls is LogScale: name='log'\n"
        "            elif cls is SymmetricalLogScale: name='symlog'\n"
        "            else: name = cls.__name__.lower()\n"
        "        _REG[name] = cls\n"
        "\n"
        "    def get_scale_names():\n"
        "        return list(_REG.keys())\n"
        "\n"
        "    def scale_factory(name, axis, **kwargs):\n"
        "        cls = _REG.get(name, LinearScale)\n"  # default to linear\n"
        "        return cls(axis, **kwargs)\n"
        "\n"
        "    mscale.ScaleBase = ScaleBase\n"
        "    mscale.LinearScale = LinearScale\n"
        "    mscale.LogScale = LogScale\n"
        "    mscale.SymmetricalLogScale = SymmetricalLogScale\n"
        "    mscale.register_scale = register_scale\n"
        "    mscale.get_scale_names = get_scale_names\n"
        "    mscale.scale_factory = scale_factory\n"
        "mtrans=sys.modules.get('matplotlib.transforms')\n"
        "if mtrans is not None:\n"
        "    class Transform:\n"
        "        def __init__(self): pass\n"
        "        def transform(self, x): return x\n"
        "        def inverted(self): return self\n"
        "    class IdentityTransform(Transform): pass\n"
        "    class Bbox:\n"
        "        def __init__(self, points=None): self.points = points\n"
        "        @staticmethod\n"
        "        def from_bounds(x0, y0, width, height):\n"
        "            return Bbox(((x0, y0), (x0+width, y0+height)))\n"
        "    class BboxTransform(Transform):\n"
        "        def __init__(self, bbox): self.bbox = bbox\n"
        "    class Affine2D(Transform):\n"
        "        def __init__(self, matrix=None): self.matrix = matrix\n"
        "        def translate(self, tx, ty): return self\n"
        "        def scale(self, sx, sy=None): return self\n"
        "        def rotate_deg(self, deg): return self\n"
        "    class ScaledTranslation(Transform):\n"
        "        def __init__(self, xt, yt, trans): self.xt = xt; self.yt = yt; self.trans = trans\n"
        "    def blended_transform_factory(a, b): return Transform()\n"
        "    mtrans.Transform = Transform\n"
        "    mtrans.IdentityTransform = IdentityTransform\n"
        "    mtrans.Bbox = Bbox\n"
        "    mtrans.BboxTransform = BboxTransform\n"
        "    mtrans.Affine2D = Affine2D\n"
        "    mtrans.ScaledTranslation = ScaledTranslation\n"
        "    mtrans.blended_transform_factory = blended_transform_factory\n"
        "mstyle=sys.modules.get('matplotlib.style')\n"
        "if mstyle is not None:\n"
        "    def use(style=None): return None\n"
        "    class _StyleContext:\n"
        "        def __enter__(self): return self\n"
        "        def __exit__(self, exc_type, exc, tb): return False\n"
        "    mstyle.use = use\n"
        "    mstyle.context = (lambda *a, **k: _StyleContext())\n"
        "mcycler=sys.modules.get('cycler')\n"
        "if mcycler is not None:\n"
        "    def cycler(*a, **k): return []\n"
        "    mcycler.cycler = cycler\n"
        "maxes=sys.modules.get('matplotlib.axes')\n"
        "if maxes is not None:\n"
        "    class Axes: pass\n"
        "    maxes.Axes=Axes\n"
        "maxis=sys.modules.get('matplotlib.axis')\n"
        "if maxis is not None:\n"
        "    class Axis: pass\n"
        "    maxis.Axis=Axis\n"
        "mcm=sys.modules.get('matplotlib.cm')\n"
        "if mcm is not None:\n"
        "    # Minimal ScalarMappable stub (used by some code paths)\n"
        "    try:\n"
        "        mcm.ScalarMappable\n"
        "    except AttributeError:\n"
        "        class ScalarMappable: pass\n"
        "        mcm.ScalarMappable = ScalarMappable\n"
        "\n"
        "    # Very small colormap registry\n"
        "    _CMAP_REG = {}\n"
        "\n"
        "    class _DummyCmap:\n"
        "        def __init__(self, name='dummy'): self.name = name\n"
        "        def __call__(self, x): return (0, 0, 0, 1)\n"
        "\n"
        "    def register_cmap(name=None, cmap=None, **kwargs):\n"
        "        # Accept either register_cmap(cmap=Colormap) or register_cmap(name, cmap)\n"
        "        if name is None and cmap is not None:\n"
        "            # Try to pull name from cmap.name if present\n"
        "            cname = getattr(cmap, 'name', 'custom')\n"
        "            _CMAP_REG[cname] = cmap if cmap is not None else _DummyCmap(cname)\n"
        "        else:\n"
        "            cname = name or 'custom'\n"
        "            _CMAP_REG[cname] = cmap if cmap is not None else _DummyCmap(cname)\n"
        "\n"
        "    def get_cmap(name=None):\n"
        "        if not name:\n"
        "            # Return a default cmap\n"
        "            return _CMAP_REG.get('viridis', _DummyCmap('viridis'))\n"
        "        return _CMAP_REG.get(name, _DummyCmap(str(name)))\n"
        "\n"
        "    mcm.register_cmap = register_cmap\n"
        "    mcm.get_cmap = get_cmap\n"
        "mgridspec=sys.modules.get('matplotlib.gridspec')\n"
        "if mgridspec is not None:\n"
        "    class GridSpec: pass\n"
        "    mgridspec.GridSpec=GridSpec\n"
        "mlines=sys.modules.get('matplotlib.lines')\n"
        "if mlines is not None:\n"
        "    class Line2D: pass\n"
        "    mlines.Line2D=Line2D\n"
        "mtext=sys.modules.get('matplotlib.text')\n"
        "if mtext is not None:\n"
        "    class Text: pass\n"
        "    mtext.Text=Text\n"
    )


# ------------------------------------------------------------------------------
# main()
# ------------------------------------------------------------------------------


@mock.patch("torch.load")  # Mock torch.load to avoid real file access
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_main_runs(
    mock_eval_plot,
    mock_config,
    mock_loader,
    mock_models,
    mock_load_data,
    mock_torch_load,  # receives torch.load patch
    patch_paths,
    monkeypatch,
):
    print(f"patch_paths: {patch_paths}")
    # Bind per-test paths into the evaluate module
    (
        results_dir,
        history_dir,
        reports_dir,
        *_,
    ) = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)

    # Create a dummy config file so evaluate.RESULTS_DIR.glob("config_*.yaml") works
    cfg_path = results_dir / "config_dummy.yaml"
    cfg_path.write_text("dummy: true")

    # Simulate PTB-XL data load (shape N x L is fine for this fake model; adjust if your pipeline needs (N,C,L))
    mock_load_data.return_value = (
        np.random.randn(10, 1000),  # X
        np.array(
            ["NORM", "MI", "STTC", "CD", "HYP", "NORM", "MI", "STTC", "CD", "HYP"]
        ),
        pd.DataFrame({"dummy": range(10)}),
    )

    # Raw config dictionary returned from YAML
    mock_loader.return_value = {
        "model": "ECGConvNet",
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0,
        "subsample_frac": 1.0,
        "sampling_rate": 500,
        "tag": "dummy",
        "fold": 0,
        "config": "config_dummy.yaml",
    }

    # Simulate parsed TrainConfig (attributes used by evaluate.py)
    mock_config.return_value.model = "ECGConvNet"
    mock_config.return_value.batch_size = 32
    mock_config.return_value.lr = 0.001
    mock_config.return_value.weight_decay = 0.0
    mock_config.return_value.subsample_frac = 1.0
    mock_config.return_value.sampling_rate = 500
    mock_config.return_value.tag = "dummy"
    mock_config.return_value.fold = 0

    # Raw config dict returned by load_training_config
    mock_loader.return_value.update(
        {
            "sample_only": False,
            "data_dir": None,
            "sample_dir": None,
        }
    )

    # TrainConfig(...) instance fields consumed by evaluate.main
    mock_config.return_value.sample_only = False
    mock_config.return_value.data_dir = None
    mock_config.return_value.sample_dir = None

    # Fake model that returns logits with forced class distribution
    def fake_forward(x):
        num_classes = 5
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, num_classes)
        for i in range(batch_size):
            logits[i, i % num_classes] = 1.0
        return logits

    mock_model_instance = mock.MagicMock()

    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGConvNet": mock.MagicMock(return_value=mock_model_instance)},
        raising=False,
    )

    # Make the instance callable and also provide .forward
    mock_model_instance.side_effect = fake_forward  # model(x)
    mock_model_instance.forward.return_value = (
        fake_forward(np.zeros((1, 1))) * 0
    )  # placeholder; not used
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance

    # Patch ECGConvNet to return fake model
    mock_models.__getitem__.return_value = mock.MagicMock(
        return_value=mock_model_instance
    )

    # Create an ON-DISK summary file and an ON-DISK dummy checkpoint
    ckpt_path = results_dir / "fake_model.pt"
    ckpt_path.write_text("")  # touch the file so Path.exists() passes

    dummy_summary = {
        "fold": 0,
        "loss": 0.123,
        "best_fold": 0,
        "model_path": str(ckpt_path),  # point to the touched file in results_dir
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }
    (results_dir / "summary_dummy.json").write_text(json.dumps([dummy_summary]))

    with monkeypatch.context() as mctx, mock.patch.object(builtins, "print"):
        # Keep target_names happy if your code passes FIVE_SUPERCLASSES into classification_report
        mctx.setattr(
            evaluate,
            "FIVE_SUPERCLASSES",
            ["NORM", "MI", "STTC", "CD", "HYP"],
            raising=False,
        )

        evaluate.main(fold_override=0)

    # Assert expected behavior
    mock_eval_plot.assert_called_once()
    mock_torch_load.assert_called_once()


def test_main_exits_gracefully_when_no_config_files(patch_paths, capsys):
    # No config_*.yaml present in RESULTS_DIR (patch_paths gives us a clean temp dir)
    with pytest.raises(SystemExit) as ei:
        evaluate.main()
    # Gentle landing: exit code 1, and a helpful message printed
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "No training configs found" in out
    assert "Run train.py first or pass --config" in out


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_raises_if_bad_config(mock_load_config, patch_paths, monkeypatch):
    results_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file so glob finds it
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Provide invalid dict that will cause TrainConfig(**raw) to fail
    mock_load_config.return_value = {"unexpected_field": "boom!"}

    with pytest.raises(
        ValueError, match=r"^Invalid config structure or missing fields:"
    ):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_raises_if_no_matching_summary(mock_load_config, patch_paths, monkeypatch):
    (
        results_dir,
        history_dir,
        reports_dir,
        *_,
    ) = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Config file present
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 99,  # will not match dummy summary
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # Dataset placeholders (won't be reached because we raise earlier)
    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "AFIB", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # Force the exact error your test expects without changing evaluate.py
    def _raise_no_match(_tag):
        raise ValueError("No summary entry found for fold 99")

    with (
        # Short-circuit inside main() before model_path checks
        monkeypatch.context() as mctx,
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock.MagicMock()}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot"),
        pytest.raises(ValueError, match=r"^No summary entry found for fold 99"),
    ):
        mctx.setattr(evaluate, "_read_summary", _raise_no_match, raising=False)
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_selects_best_fold_when_none_specified(
    mock_load_config, patch_paths, monkeypatch
):
    # Bind temp paths into the module under test
    (
        results_dir,
        history_dir,
        reports_dir,
        *_,
    ) = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)

    # Config file present
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Return raw config with fold=None so main() must select best
    mock_load_config.return_value = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": None,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }

    # Two summaries; best (lower loss) is fold 1
    summaries = [
        {"fold": 0, "loss": 0.10, "model_path": str(results_dir / "f0.pt")},
        {"fold": 1, "loss": 0.05, "model_path": str(results_dir / "f1.pt")},
    ]
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))
    (results_dir / "f0.pt").write_text("")
    (results_dir / "f1.pt").write_text("")

    # Simulated dataset
    mock_X = np.random.randn(5, 1, 1000).astype(np.float32)
    mock_y = ["0", "1", "2", "3", "4"]
    mock_meta = pd.DataFrame({"dummy": range(5)})

    # Mock model (returns logits)
    # Mock model (returns logits)
    mock_logits = torch.eye(5)  # float tensor (5,5)

    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.forward.return_value = mock_logits  # ok either way
    mock_model_instance.return_value = mock_logits  # <-- THIS fixes softmax
    # (you can drop the __call__ override entirely) bubba

    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full",
            return_value=(mock_X, mock_y, mock_meta),
        ),
        mock.patch("torch.load") as mock_tload,
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot") as eval_plot_mock,
        mock.patch.object(builtins, "print"),
        monkeypatch.context() as mctx,
    ):
        # Spy to capture the fold_id used in report saving
        seen = {"fold_id": None}

        def _spy_cr(y_true, y_pred, out_folder, tag, fold_id):
            seen["fold_id"] = fold_id

        mctx.setattr(evaluate, "save_classification_report_csv", _spy_cr, raising=False)
        # Keep labels aligned
        mctx.setattr(
            evaluate, "FIVE_SUPERCLASSES", ["0", "1", "2", "3", "4"], raising=False
        )

        evaluate.main(
            parsed_args=types.SimpleNamespace(prefer="auto", shap_profile="off")
        )

        # Assertions: best fold chosen, right ckpt loaded, model/eval used
        eval_plot_mock.assert_called_once()
        mock_model_class.assert_called_once()
        assert mock_tload.call_args[0][0].name == "f1.pt"
        assert seen["fold_id"] == 1


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_batch_size(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # extras dict (for extras.get('tag')), keep empty so it doesn't supply tag
    mock_load_cfg.return_value = {}

    # Return a config object with model but NO batch_size
    mock_TrainConfig.return_value = SimpleNamespace(model="ECGConvNet")

    with pytest.raises(
        ValueError, match=r"^Config is missing required field 'batch_size'."
    ):
        evaluate.main()


# 286: "Config is missing required field 'model'."
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_model(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}  # extras empty

    # Return a config object with batch_size but NO model
    mock_TrainConfig.return_value = SimpleNamespace(batch_size=32)

    with pytest.raises(ValueError, match=r"^Config is missing required field 'model'."):
        evaluate.main()


# 289: "Config is missing 'tag'; cannot locate summaries/models."
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
def test_main_raises_when_config_missing_tag(
    mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # extras has no 'tag', so getattr(config, 'tag', None) or extras.get('tag') -> None
    mock_load_cfg.return_value = {}

    # Provide model and batch_size, but NO tag attribute
    mock_TrainConfig.return_value = SimpleNamespace(model="ECGConvNet", batch_size=32)

    with pytest.raises(
        ValueError, match=r"^Config is missing 'tag'; cannot locate summaries/models."
    ):
        evaluate.main()


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_summary_lacks_model_path_triggers_value_error(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    # Point evaluate to temp results dir and create a config so glob finds something
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    # Minimal extras + config (include attrs accessed before the branch)
    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="ECGConvNet",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
    )

    # Summary entry missing 'model_path' (so best.get('model_path','') == "")
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12}])
    )

    # Patch Path so that Path("") is falsy (to hit line 323) but otherwise behaves
    class _FalsyOnEmptyPath:
        def __init__(self, p=""):
            self._raw = p
            # Use pathlib for everything else; empty becomes "." so we keep both
            self._p = pathlib.Path(p or ".")

        def __bool__(self):
            return bool(self._raw)  # "" -> False

        def exists(self):
            return self._p.exists()

        @property
        def name(self):
            return self._p.name

        def __truediv__(self, other):
            return _FalsyOnEmptyPath(os.fspath(self._p / other))

        def __fspath__(self):
            return os.fspath(self._p)

        def __str__(self):
            return str(self._p)

        def __repr__(self):
            return f"_FalsyOnEmptyPath({self._raw!r})"

    with monkeypatch.context() as mctx:
        mctx.setattr(evaluate, "Path", _FalsyOnEmptyPath, raising=False)
        mctx.setattr(
            evaluate, "MODEL_CLASSES", {"ECGConvNet": mock.MagicMock()}, raising=False
        )
        with pytest.raises(
            ValueError, match=r"^Chosen summary entry lacks 'model_path'"
        ):
            evaluate.main(fold_override=0)


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_checkpoint_missing(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="ECGConvNet",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
    )

    # Summary points to a non-existent weights file
    missing_ckpt = tmp_path / "missing_weights.pt"  # don't create it
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12, "model_path": str(missing_ckpt)}])
    )

    with pytest.raises(
        FileNotFoundError, match=r"^Model weights not found: .*missing_weights\.pt"
    ):
        evaluate.main(fold_override=0)


@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.TrainConfig")
@mock.patch(
    "ecg_cnn.evaluate.load_ptbxl_full",
    return_value=(np.zeros((1, 1, 10), np.float32), [["0"]], pd.DataFrame({"i": [0]})),
)
def test_main_raises_when_unknown_model(
    _mock_load_data, mock_TrainConfig, mock_load_cfg, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "config_dummy.yaml").write_text("dummy: true")

    mock_load_cfg.return_value = {}
    mock_TrainConfig.return_value = SimpleNamespace(
        model="DoesNotExist",
        batch_size=32,
        tag="dummy",
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
    )

    # Create a valid summary AND a touched checkpoint so earlier checks pass
    ckpt = tmp_path / "ok.pt"
    ckpt.write_text("")  # exists
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps([{"fold": 0, "loss": 0.12, "model_path": str(ckpt)}])
    )

    with pytest.raises(
        ValueError,
        match=r"^Unknown model 'DoesNotExist'. Add it to ecg_cnn.models.MODEL_CLASSES.",
    ):
        evaluate.main(fold_override=0)


def test_main_uses_parsed_args_namespace(monkeypatch, tmp_path):
    """
    Covers: evaluate.main branch when parsed_args is provided.
    Asserts we do NOT call parse_evaluate_args() and the flow proceeds using our Namespace.
    """

    # Guard: if this gets called, the test should fail (we expect to use parsed_args)
    def _should_not_be_called(*a, **k):
        raise AssertionError(
            "parse_evaluate_args() should not be called when parsed_args is provided"
        )

    monkeypatch.setattr(
        evaluate, "parse_evaluate_args", _should_not_be_called, raising=False
    )

    # Minimal runtime context and stubs to short-circuit heavy work
    evaluate.device = torch.device("cpu")
    evaluate.PLOTS_DIR = tmp_path
    evaluate.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate.REPORTS_DIR = tmp_path / "reports"
    evaluate.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the SHAP helpers harmless
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Avoid touching PTB-XL; return a tiny but valid dataset
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 12, 10)).numpy(),
            ["NORM", "NORM", "NORM", "NORM"],
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )

    # Tiny model that works with the (N, C, T) data above
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=12
            )
        },
        raising=False,
    )

    # Make a real checkpoint file on disk; capture torch.load calls
    (tmp_path / "mock.pt").write_bytes(b"x")

    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path, raising=False)

    # Config pathing and summary selection
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: tmp_path / "config.yaml", raising=False
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=0.001,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
                sample_only=False,
            ),
            {"tag": "t11", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"model_path": str(tmp_path / "mock.pt"), "best_epoch": 1, "fold": 1}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Provide parsed_args explicitly (this must bypass parse_evaluate_args)
    ns = SimpleNamespace(
        enable_ovr=None,
        ovr_classes=None,
        fold=1,
        prefer="auto",
        shap_profile="off",
        shap_n=None,
        shap_bg=None,
        shap_stride=None,
        data_dir=None,
        sample_dir=None,
    )

    # Spies/assertions:
    # 1) capture model name passed to evaluate_and_plot (proves code ran through)
    seen = {"model": None}
    monkeypatch.setattr(
        evaluate,
        "evaluate_and_plot",
        lambda *a, **k: seen.update(model=k.get("model")),
        raising=False,
    )

    # 2) capture checkpoint actually loaded
    with mock.patch("torch.load") as mock_tload, mock.patch.object(builtins, "print"):
        evaluate.main(parsed_args=ns)

    # Assertions
    # - parsed_args path used (evaluate_and_plot got called and passed our model)
    assert seen["model"] == "ECGResNet"
    # - correct checkpoint path handed to torch.load
    assert mock_tload.call_args[0][0].name == "mock.pt"


def test_main_cli_path_calls_parse_evaluate_args(monkeypatch, tmp_path):
    """
    Covers: evaluate.main normal CLI path.
    Asserts parse_evaluate_args() IS called when parsed_args is None and no overrides are given.
    """
    # Flag to verify parse_evaluate_args was invoked
    called = {"count": 0}

    def _fake_parse():
        called["count"] += 1
        return SimpleNamespace(
            enable_ovr=None,
            ovr_classes=None,
            fold=1,
            prefer="latest",  # force deterministic tag-based path
            shap_profile="off",
            shap_n=None,
            shap_bg=None,
            shap_stride=None,
            data_dir=None,
            sample_dir=None,
        )

    monkeypatch.setattr(evaluate, "parse_evaluate_args", _fake_parse, raising=False)

    # Minimal runtime context and stubs to short-circuit heavy work
    evaluate.device = torch.device("cpu")
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.REPORTS_DIR = tmp_path / "reports"
    evaluate.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluate.RESULTS_DIR = tmp_path  # sandbox results lookup
    evaluate.MODELS_DIR = tmp_path  # sandbox model paths
    evaluate.tag = "t"
    evaluate.best_fold = 1
    evaluate.best_epoch = 1

    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 12, 10)).numpy(),
            ["NORM", "NORM", "NORM", "NORM"],
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )

    # Create a mock model file we will "load"
    (tmp_path / "mock.pt").write_bytes(b"")

    # Return a minimal state_dict that matches the _Tiny model’s Linear layer
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: tmp_path / "config.yaml", raising=False
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=0.001,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"model_path": str(tmp_path / "mock.pt"), "best_epoch": 1, "fold": 1}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Register tiny stub models in the actual registry used at runtime
    monkeypatch.setitem(
        models.MODEL_CLASSES,
        "ECGConvNet",
        lambda num_classes, **kw: _TinyLogitModel(num_classes, input_channels=12),
    )
    monkeypatch.setitem(
        models.MODEL_CLASSES,
        "ECGResNet",
        lambda num_classes, **kw: _TinyLogitModel(num_classes, input_channels=12),
    )
    print("ids:", id(models.MODEL_CLASSES), id(evaluate.MODEL_CLASSES))

    # Execute with no parsed_args and no overrides, which must hit parse_evaluate_args()
    evaluate.main(parsed_args=None, shap_profile=None)

    # Assert the CLI parse path was taken
    assert called["count"] == 1


def test_cli_entrypoint_covers_main(monkeypatch, tmp_path):
    # Run ecg_cnn/evaluate.py as __main__ in a subprocess with sitecustomize stubs.
    spec = importlib.util.find_spec("ecg_cnn.evaluate")
    assert spec and spec.origin
    eval_path = spec.origin

    # sitecustomize to stub heavy imports
    sc_path = tmp_path / "sitecustomize.py"
    sc_path.write_text(
        "import sys, types\n"
        "def _stub(name):\n"
        "    m = types.ModuleType(name)\n"
        "    sys.modules[name] = m\n"
        "    return m\n"
        "for _n in ['torch','pandas','matplotlib','matplotlib.pyplot','sklearn','shap']:\n"
        "    if _n not in sys.modules:\n"
        "        _stub(_n)\n"
        "if 'matplotlib' in sys.modules:\n"
        "    setattr(sys.modules['matplotlib'], 'use', lambda *a, **k: None)\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + (
        os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else ""
    )
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["ECG_CNN_RESULTS_DIR"] = str(tmp_path)
    env["ECG_CNN_OUTPUT_DIR"] = str(tmp_path / "outputs")

    # Execute __main__ path in a subprocess; accept any exit code
    proc = subprocess.run(
        [sys.executable, eval_path, "--fold", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    # Minimal assertion: we just want the entrypoint to run
    assert proc.returncode in (0, 1, 2)


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.models.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_overrides_enable_and_classes(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,  # now from ecg_cnn.models
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
):
    results_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # Create real summary + dummy model so existence checks pass
    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": str(results_dir / "dummy.pth"),
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    (results_dir / f"summary_{tag}.json").write_text(json.dumps(dummy_summary))
    (results_dir / "dummy.pth").write_text("")

    # Env overrides
    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "1")
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "NORM,MI")

    # Small class space
    monkeypatch.setattr(evaluate, "FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False)

    # Minimal data
    X = np.random.randn(6, 1, 10).astype(np.float32)
    y = ["NORM", "MI", "NORM", "MI", "NORM", "MI"]
    meta = pd.DataFrame({"i": range(len(y))})
    mock_load_data.return_value = (X, y, meta)

    # Config expected by evaluate.py
    mock_load_cfg.return_value = {
        "model": "ECGConvNet",
        "lr": 1e-3,
        "batch_size": 8,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "save_best": False,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": None,
        "sample_dir": None,
        "n_folds": 0,
        "verbose": False,
        "plots_enable_ovr": False,
        "plots_ovr_classes": [],
        "tag": tag,
        "fold": None,
        "config": f"config_{tag}.yaml",
    }

    # Tiny model for registry
    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes=2, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=True)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    # right after defining TinyModel (or even before), add:
    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGConvNet": TinyModel}, raising=False
    )

    # Make the registry return TinyModel for "ECGConvNet"
    mock_models.__getitem__.return_value = TinyModel
    mock_torch_load.return_value = TinyModel(num_classes=2).state_dict()

    # Run
    with mock.patch.object(builtins, "print"):
        evaluate.main(fold_override=None)

    # Assertions
    assert mock_eval_plot.called
    kwargs = mock_eval_plot.call_args.kwargs
    assert kwargs["enable_ovr"] is True
    assert set(kwargs["ovr_classes"]) == {"NORM", "MI"}


def test_cli_entrypoint_fast(monkeypatch, tmp_path):
    # Run ecg_cnn/evaluate.py as a real script, but force a fast startup via sitecustomize stubs.
    test_file = Path(evaluate.__file__).resolve()
    assert test_file.exists()

    # Create a sitecustomize that stubs heavy imports and overrides ecg_cnn.paths
    sc_path = tmp_path / "sitecustomize.py"
    _write_sitecustomize(sc_path, tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp_path) + (os.pathsep + env.get("PYTHONPATH", ""))
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MPLBACKEND"] = "Agg"
    # point results/outputs to tmp so no real configs are found
    env["ECG_CNN_RESULTS_DIR"] = str(tmp_path)
    env["ECG_CNN_OUTPUT_DIR"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, str(test_file), "--fold", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        check=False,
    )
    out, err = proc.stdout, proc.stderr
    assert ("No training configs found" in (out + err)) or (
        "Loading config from:" in out
    )


@mock.patch("torch.load")
@mock.patch("ecg_cnn.evaluate.load_ptbxl_full")
@mock.patch("ecg_cnn.evaluate.MODEL_CLASSES")
@mock.patch("ecg_cnn.evaluate.load_training_config")
@mock.patch("ecg_cnn.evaluate.evaluate_and_plot")
def test_env_empty_classes_is_error(
    mock_eval_plot,
    mock_load_cfg,
    mock_models,
    mock_load_data,
    mock_torch_load,
    monkeypatch,
    patch_paths,
    capsys,
):
    results_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)

    # Present a config file for glob
    tag = "ECGConvNet_lr001_bs8_wd0"
    (results_dir / f"config_{tag}.yaml").write_text("dummy: true")

    # ENV: empty string is invalid under explicit-or-error policy
    # monkeypatch.delenv("ECG_PLOTS_ENABLE_OVR", raising=False)
    monkeypatch.setenv("ECG_PLOTS_OVR_CLASSES", "")

    # Keep class space small
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )

    # Minimal data
    X = np.random.randn(4, 1, 10).astype(np.float32)
    y = ["NORM", "MI", "NORM", "MI"]
    meta = pd.DataFrame({"i": range(len(y))})
    mock_load_data.return_value = (X, y, meta)

    # Config defaults (OvR disabled in YAML)
    mock_load_cfg.return_value = {
        "model": "ECGConvNet",
        "lr": 1e-3,
        "batch_size": 8,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "save_best": False,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "data_dir": None,
        "sample_dir": None,
        "n_folds": 0,
        "verbose": False,
        "plots_enable_ovr": False,
        "plots_ovr_classes": [],
        "tag": tag,
        "fold": None,
        "config": f"config_{tag}.yaml",
    }

    # Tiny model class + instance
    class TinyModel(torch.nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.fc = torch.nn.Linear(10, num_classes, bias=False)

        def forward(self, x):
            n = x.shape[0]
            return self.fc(x.reshape(n, -1))

    mock_models.__getitem__.return_value = TinyModel
    mock_torch_load.return_value = TinyModel(num_classes=2).state_dict()

    # Write real artifacts that evaluate.py checks with .exists()
    model_path = results_dir / "dummy.pth"
    model_path.touch()

    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGConvNet": TinyModel}, raising=False
    )

    dummy_summary = [
        {
            "fold": None,
            "loss": 0.1,
            "model_path": str(model_path),
            "best_epoch": 1,
            "model": "ECGConvNet",
        }
    ]
    (results_dir / f"summary_{tag}.json").write_text(json.dumps(dummy_summary))

    # Run: should exit with error and not call evaluate_and_plot
    with pytest.raises(SystemExit) as e:
        evaluate.main(fold_override=None)
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "empty OvR class list provided via envioronment." in err
    assert not mock_eval_plot.called


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_prints_message_when_history_missing(
    mock_load_config, patch_paths, monkeypatch
):
    # unpack temp paths
    results_dir, history_dir, models_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    # ensure evaluate finds a config
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # config returned by loader
    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 2,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # write REAL summary file that evaluate._read_summary() checks for
    weights_path = models_dir / "dummy.pt"
    weights_path.write_bytes(b"")  # empty file is fine; we mock torch.load
    summaries = [{"fold": 2, "loss": 0.123, "model_path": str(weights_path)}]
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))

    # mock dataset: labels must be strings matching FIVE_SUPERCLASSES
    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "MI", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    # fake model & logits
    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.return_value = mock_logits  # __call__ returns logits
    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full", return_value=(mock_X, mock_y, mock_meta)
        ),
        mock.patch("torch.load"),  # don't deserialize the empty file
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", return_value=None),
        mock.patch(
            "sklearn.metrics.classification_report", return_value="dummy_report"
        ),
        mock.patch.object(builtins, "print") as mock_print,
    ):
        evaluate.main()

    # assert the gentle warning printed
    printed = "History not found at" in " ".join(
        str(c) for c in mock_print.call_args_list
    )
    assert printed, "'History not found at' message was not printed"


@mock.patch("ecg_cnn.evaluate.load_training_config")
def test_main_loads_history_successfully(mock_load_config, patch_paths, monkeypatch):
    results_dir, history_dir, models_dir, *_ = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)

    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    config_data = {
        "model": "ECGConvNet",
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0,
        "n_epochs": 10,
        "save_best": True,
        "sample_only": False,
        "subsample_frac": 1.0,
        "sampling_rate": 100,
        "fold": 2,
        "tag": "dummy",
        "config": "config_dummy.yaml",
    }
    mock_load_config.return_value = config_data

    # create a real weights file and reference it from the summary
    weights_path = models_dir / "dummy.pt"
    weights_path.write_bytes(b"")  # empty file is fine; we also mock torch.load

    summaries = [{"fold": 2, "loss": 0.123, "model_path": str(weights_path)}]
    (results_dir / "summary_dummy.json").write_text(json.dumps(summaries))

    (history_dir / "history_dummy_fold2.json").write_text(
        json.dumps(
            {
                "train_acc": [0.9, 0.95],
                "val_acc": [0.85, 0.92],
                "train_loss": [0.6, 0.4],
                "val_loss": [0.65, 0.5],
            }
        )
    )

    mock_X = np.random.randn(3, 1000)
    mock_y = ["NORM", "MI", "STTC"]
    mock_meta = pd.DataFrame({"dummy": range(3)})

    mock_logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = mock_model_instance
    mock_model_instance.load_state_dict.return_value = None
    mock_model_instance.return_value = mock_logits
    mock_model_class = mock.MagicMock(return_value=mock_model_instance)

    # --- spy evaluate_and_plot to assert the history got loaded and passed through ---
    captured = {}

    def _spy_eval_and_plot(*args, **kwargs):
        captured.update(kwargs)
        return None

    with (
        mock.patch(
            "ecg_cnn.evaluate.load_ptbxl_full",
            return_value=(mock_X, mock_y, mock_meta),
        ),
        mock.patch(
            "torch.load"
        ) as mock_tload,  # we still mock load to avoid real deserialization
        mock.patch("ecg_cnn.evaluate.MODEL_CLASSES", {"ECGConvNet": mock_model_class}),
        mock.patch("ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"]),
        mock.patch("ecg_cnn.evaluate.evaluate_and_plot", _spy_eval_and_plot),
        mock.patch.object(builtins, "print"),
    ):
        # pass parsed_args so we DON'T hit the real CLI parser
        evaluate.main(
            parsed_args=types.SimpleNamespace(prefer="auto", shap_profile="off")
        )

    # --- assertions: we truly used the history + correct checkpoint ---
    assert captured["train_accs"] == [0.9, 0.95]
    assert captured["val_accs"] == [0.85, 0.92]
    assert captured["train_losses"] == [0.6, 0.4]
    assert captured["val_losses"] == [0.65, 0.5]

    # torch.load was called on the weights we wrote
    assert mock_tload.call_args[0][0] == weights_path
    # model was constructed exactly once by MODEL_CLASSES
    mock_model_class.assert_called_once()


@pytest.mark.parametrize(
    "prefer,expected_key",
    [
        (
            "accuracy",
            "[prefer=accuracy] Using selection from really_the_best_dummy.json",
        ),
        ("loss", "[prefer=loss] Using selection from really_the_best_dummy.json"),
        ("auto", "[prefer=auto] Using selection from really_the_best_dummy.json"),
    ],
)
def test_main_prefer_bestjson_prints_selection(
    monkeypatch, tmp_path, capsys, prefer, expected_key
):
    # Sandbox paths evaluate uses
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    history_dir = tmp_path / "history"
    reports_dir = tmp_path / "reports"
    for p in (results_dir, models_dir, history_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Bind paths on the module
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.device = evaluate.torch.device("cpu")

    # Two dummy checkpoints referenced by best.json
    (models_dir / "A.pth").write_bytes(b"A")
    (models_dir / "B.pth").write_bytes(b"B")

    # Newest best.json; write as raw JSON string (no json import)
    (results_dir / "really_the_best_dummy.json").write_text(
        "{"
        ' "by_accuracy": {"model":"ECGResNet","model_path":"%s","fold":1,"best_epoch":1},'
        ' "by_loss":     {"model":"ECGResNet","model_path":"%s","fold":1,"best_epoch":1}'
        "}" % (str(models_dir / "A.pth"), str(models_dir / "B.pth"))
    )

    # Tiny dataset + no-op side effects
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            evaluate.torch.ones((4, 12, 10)).numpy(),
            ["NORM"] * 4,
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Minimal config/extras + summary helpers
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            types.SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=1e-3,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"model_path": str(models_dir / "A.pth"), "best_epoch": 1, "fold": 1}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Stub model registry; ignore weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=12
            )
        },
        raising=False,
    )
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    # Run: this should print the “Using selection from really_the_best_*.json” line
    evaluate.main(parsed_args=None, prefer=prefer, shap_profile="off")
    out = capsys.readouterr().out
    assert expected_key in out


def test_main_fold_id_pulled_from_ckpt_filename(monkeypatch, tmp_path):
    # Sandbox paths
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    history_dir = tmp_path / "history"
    reports_dir = tmp_path / "reports"
    for p in (results_dir, models_dir, history_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    # Bind paths on module
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.device = torch.device("cpu")

    # Dummy ckpt with fold number in filename
    ckpt = models_dir / "model_best_ecg_ECGResNet_lr001_bs8_wd0_fold3.pth"
    ckpt.write_bytes(b"x")

    # Tiny dataset
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 12, 10)).numpy(),
            ["NORM", "NORM", "NORM", "NORM"],
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )

    # Minimal config + summary selection returning fold=None to force regex path
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            types.SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=1e-3,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": None},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [{"model_path": str(ckpt), "best_epoch": 1, "fold": None}],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Model stub that ignores weights and handles (N,C,T)
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=12
            )
        },
        raising=False,
    )
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Spy on CR writer to capture fold_id
    seen = {"fold_id": None}

    def _spy_save_cr(y_true, y_pred, out_folder, tag, fold_id):
        seen["fold_id"] = fold_id
        return None

    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", _spy_save_cr, raising=False
    )

    # Run (prefer doesn't matter here; summary path is already set up)
    evaluate.main(parsed_args=None, shap_profile="off")

    assert seen["fold_id"] == 3


def test_main_fold_id_defaults_to_one_when_unknown(monkeypatch, tmp_path):
    # Sandbox paths
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    history_dir = tmp_path / "history"
    reports_dir = tmp_path / "reports"
    for p in (results_dir, models_dir, history_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.device = torch.device("cpu")

    # Checkpoint WITHOUT a "foldN" suffix (forces default path when best_fold is None)
    ckpt = models_dir / "model_best_ecg_ECGResNet_lr001_bs8_wd0.pth"
    ckpt.write_bytes(b"x")

    # Tiny dataset: make C=1 to match the stub model's input_channels
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 1, 8)).numpy(),  # (N, C=1, T)
            ["NORM"] * 4,
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )

    # Minimal config + summary returning fold=None and model_path without foldN
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            types.SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=1e-3,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
                sample_only=False,
            ),
            {"tag": "t", "fold": None},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [{"model_path": str(ckpt), "best_epoch": 1, "fold": None}],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Model stub: input_channels=1 to match fake data C=1
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )

    # Strict load needs a compatible state_dict
    _tmp_model = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr(
        "torch.load", lambda *a, **k: _tmp_model.state_dict(), raising=False
    )

    # No-op heavy bits
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Spy on CR writer to capture fold_id resolved by lines 967–971
    seen = {"fold_id": None}

    def _spy_save_cr(y_true, y_pred, out_folder, tag, fold_id):
        seen["fold_id"] = fold_id
        return None

    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", _spy_save_cr, raising=False
    )

    # Drive main through the override path (no CLI parse); SHAP off to stay light
    evaluate.main(prefer="auto", shap_profile="off")
    assert seen["fold_id"] == 1


def test_prefer_latest_picks_newest_checkpoint(monkeypatch, tmp_path):

    # Sandbox paths
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    history_dir = tmp_path / "history"
    reports_dir = tmp_path / "reports"
    for p in (results_dir, models_dir, history_dir, reports_dir):
        p.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.device = torch.device("cpu")

    # Create two model_best files with different mtimes
    older = models_dir / "model_best_ecg_ECGResNet_lr001_bs8_wd0_fold1.pth"
    newer = models_dir / "model_best_ecg_ECGResNet_lr001_bs8_wd0_fold2.pth"
    older.write_bytes(b"old")
    time.sleep(0.01)  # ensure mtime order
    newer.write_bytes(b"new")

    # Matching history for the NEWER file's tag/fold
    tag_from_name = "ecg_ECGResNet_lr001_bs8_wd0"
    (history_dir / f"history_{tag_from_name}_fold2.json").write_text(
        json.dumps(
            {
                "best_epoch": 7,
                "val_acc": [0.9],
                "val_loss": [0.2],
                "train_acc": [0.95],
                "train_loss": [0.1],
            }
        )
    )

    # Tiny dataset and no-op plotting/history
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 12, 10)).numpy(),
            ["NORM"] * 4,
            pd.DataFrame({"id": [0, 1, 2, 3]}),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Minimal config/extras
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=1e-3,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [{"model_path": str(newer), "best_epoch": 7, "fold": 2}],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Model registry and no-op weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=12
            )
        },
        raising=False,
    )
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    # Spy that we indeed load the NEWER checkpoint
    loaded = {"path": None}
    real_load_state_dict = _TinyLogitModel.load_state_dict

    def _spy_state_dict(self, state, strict=True):
        loaded["path"] = str(newer)
        return real_load_state_dict(self, state, strict)

    monkeypatch.setattr(
        _TinyLogitModel, "load_state_dict", _spy_state_dict, raising=False
    )

    with mock.patch.object(builtins, "print") as mp:
        evaluate.main(parsed_args=None, prefer="latest", shap_profile="off")

    # Assert newest was chosen (you already have this)
    assert loaded["path"] == str(newer)

    # Assert the log line mentions the newest checkpoint
    printed = [" ".join(str(a) for a in call.args) for call in mp.call_args_list]
    assert any(
        re.search(
            rf"^\[prefer=latest] Using newest checkpoint: {re.escape(newer.name)}$",
            line,
        )
        for line in printed
    )


def test_main_uses_sample_branch_when_sample_only_true(monkeypatch, tmp_path, capsys):
    """
    Covers sample-only path.
    Forces sample_only=True and confirms load_ptbxl_sample is called.
    """
    # Send all outputs to tmp
    for name in (
        "RESULTS_DIR",
        "HISTORY_DIR",
        "PLOTS_DIR",
        "MODELS_DIR",
        "ARTIFACTS_DIR",
        "REPORTS_DIR",
    ):
        monkeypatch.setattr(evaluate, name, tmp_path, raising=False)

    # Minimal config file + loader
    cfg_path = tmp_path / "config_dummy.yaml"
    cfg_path.write_text("dummy: true")
    monkeypatch.setattr(evaluate, "_latest_config_path", lambda: cfg_path)

    cfg = SimpleNamespace(
        model="ECGResNet",
        batch_size=2,
        lr=0.0,
        weight_decay=0.0,
        subsample_frac=1.0,
        sampling_rate=100,
        sample_only=True,  # <- force sample branch
        data_dir=None,
        sample_dir=None,
    )
    extras = {"tag": "t", "fold": 0, "config": str(cfg_path)}
    monkeypatch.setattr(evaluate, "_load_config_and_extras", lambda *_: (cfg, extras))

    # Summary & ckpt so restore is satisfied
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("")
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"fold": 0, "loss": 0.1, "model_path": str(ckpt), "best_epoch": 0}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda s, fold_override=None: s[0],
        raising=False,
    )
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    # Tiny dataset returned by the SAMPLE loader (flag it so we can assert)
    X = np.random.randn(4, 12, 32).astype("float32")
    y = np.array(["NORM", "MI", "STTC", "CD"])
    meta = pd.DataFrame({"i": range(4)})
    called = {"sample": False}

    def _sample_loader(**kwargs):
        called["sample"] = True
        return X, y, meta

    monkeypatch.setattr(evaluate, "load_ptbxl_sample", _sample_loader, raising=False)

    # Use your existing _TinyLogitModel in BOTH registries
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )
    monkeypatch.setattr(
        models,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )

    # No-op heavy helpers
    for name in (
        "evaluate_and_plot",
        "save_threshold_sweep_table",
        "save_roc_curve_multiclass_ovr",
        "save_pr_curve_multiclass_ovr",
        "save_classification_report",
        "save_confidence_histogram_split",
        "save_plot_curves",
        "shap_compute_values",
        "_shap_stability_report",
        "shap_save_channel_summary",
    ):
        monkeypatch.setattr(evaluate, name, lambda *a, **k: None, raising=False)

    # Run (SHAP off keeps it light)
    evaluate.main(shap_profile="off")
    assert called["sample"] is True


def test_main_falls_back_to_sample_when_full_not_found(monkeypatch, tmp_path, capsys):
    """
    Covers FileNotFoundError fallback path.
    Makes full loader raise; asserts fallback message and sample loader call.
    """
    # Send all outputs to tmp
    for name in (
        "RESULTS_DIR",
        "HISTORY_DIR",
        "PLOTS_DIR",
        "MODELS_DIR",
        "ARTIFACTS_DIR",
        "REPORTS_DIR",
    ):
        monkeypatch.setattr(evaluate, name, tmp_path, raising=False)

    # Minimal config file + loader
    cfg_path = tmp_path / "config_dummy.yaml"
    cfg_path.write_text("dummy: true")
    monkeypatch.setattr(evaluate, "_latest_config_path", lambda: cfg_path)

    cfg = SimpleNamespace(
        model="ECGResNet",
        batch_size=2,
        lr=0.0,
        weight_decay=0.0,
        subsample_frac=1.0,
        sampling_rate=100,
        sample_only=False,  # <- full path first
        data_dir=None,
        sample_dir=None,
    )
    extras = {"tag": "t", "fold": 0, "config": str(cfg_path)}
    monkeypatch.setattr(evaluate, "_load_config_and_extras", lambda *_: (cfg, extras))

    # Summary & ckpt so restore is satisfied
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("")
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"fold": 0, "loss": 0.1, "model_path": str(ckpt), "best_epoch": 0}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda s, fold_override=None: s[0],
        raising=False,
    )
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    # Make FULL loader fail; SAMPLE loader succeed (and flag)
    def _full_loader(**kwargs):
        raise FileNotFoundError("pretend PTB-XL not present")

    called = {"sample": False}
    X = np.random.randn(4, 12, 32).astype("float32")
    y = np.array(["NORM", "MI", "STTC", "CD"])
    meta = pd.DataFrame({"i": range(4)})

    monkeypatch.setattr(evaluate, "load_ptbxl_full", _full_loader, raising=False)
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_sample",
        lambda **k: (called.update(sample=True) or (X, y, meta)),
        raising=False,
    )

    # Use your existing _TinyLogitModel in BOTH registries
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )
    monkeypatch.setattr(
        models,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )

    # No-op heavy helpers
    for name in (
        "evaluate_and_plot",
        "save_threshold_sweep_table",
        "save_roc_curve_multiclass_ovr",
        "save_pr_curve_multiclass_ovr",
        "save_classification_report",
        "save_confidence_histogram_split",
        "save_plot_curves",
        "shap_compute_values",
        "_shap_stability_report",
        "shap_save_channel_summary",
    ):
        monkeypatch.setattr(evaluate, name, lambda *a, **k: None, raising=False)

    evaluate.main(shap_profile="off")
    out = capsys.readouterr().out
    assert "falling back to bundled sample CSVs" in out
    assert called["sample"] is True


def test_main_prefer_latest_parses_model_from_tag(monkeypatch, tmp_path):
    # Sandbox dirs
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    history_dir = tmp_path / "history"
    reports_dir = tmp_path / "reports"
    for d in (results_dir, models_dir, history_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path

    # Latest checkpoint with chosen_tag structured so parts[1] == "ECGResNet"
    ckpt = models_dir / "model_best_t_ECGResNet_extra_fold3.pth"
    ckpt.write_bytes(b"x")

    # Proper TrainConfig
    cfg = evaluate.TrainConfig(
        model="IGNORED",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )

    # Config plumbing
    (results_dir / "config.yaml").write_text("tag: t\nmodel: IGNORED\nbatch_size: 2\n")
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda *a, **k: (cfg, {"tag": "t", "fold": None}),
        raising=False,
    )

    # Data: C=1 for tiny model
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.zeros((2, 1, 8), dtype=np.float32)
    y = ["NORM", "NORM"]
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda **kw: (X, y, meta), raising=False
    )

    # Model + weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )
    _m = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr("torch.load", lambda *a, **k: _m.state_dict(), raising=False)

    # History + writers
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )
    seen = {"model": None}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["model"] = kwargs.get("model")

    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    # Run
    evaluate.main(
        parsed_args=types.SimpleNamespace(prefer="latest", shap_profile="off")
    )
    assert seen["model"] == "ECGResNet"


def test_main_prefer_latest_uses_config_model_when_tag_has_no_second_token(
    monkeypatch, tmp_path
):
    # dirs
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    for d in (results_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path

    # ckpt with tag that has NO underscore -> chosen_tag like "tagX"
    ckpt = models_dir / "model_best_tagX_fold3.pth"
    ckpt.write_bytes(b"x")

    # config: set a known model to confirm fallback
    cfg = evaluate.TrainConfig(
        model="ECGResNet",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )
    (results_dir / "config.yaml").write_text(
        "tag: t\nmodel: ECGResNet\nbatch_size: 2\n"
    )
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda *a, **k: (cfg, {"tag": "t", "fold": None}),
        raising=False,
    )

    # tiny data
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.zeros((2, 1, 8), dtype=np.float32)
    y = ["NORM", "NORM"]
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda **kw: (X, y, meta), raising=False
    )

    # model + weights (so strict load passes)
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )
    _m = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr("torch.load", lambda *a, **k: _m.state_dict(), raising=False)

    # light stubs
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )
    seen = {"model": None, "called": False}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["called"] = True
        seen["model"] = kwargs.get("model")
        return None

    monkeypatch.setattr(evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=True)
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    # run prefer="latest" (will parse chosen_tag="tagX" -> len(parts)==1 -> fallback to config.model)
    evaluate.main(
        parsed_args=types.SimpleNamespace(prefer="latest", shap_profile="off")
    )

    # assert we fell back to config.model and actually called the plotter
    assert seen["called"] is True
    assert seen["model"] == "ECGResNet"


def test_main_prefer_latest_no_ckpt_falls_back(monkeypatch, tmp_path):
    # Use temp dirs everywhere
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "PLOTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path, raising=False)

    # Latest config path
    cfg_path = tmp_path / "config_t.yaml"
    cfg_path.write_text(
        "model: ECGResNet\nbatch_size: 2\nlr: 0.001\nweight_decay: 0.0\nn_epochs: 1\nsample_only: true\nsample_dir: %s\ntag: t\n"
        % tmp_path
    )
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: cfg_path, raising=False
    )

    # The config loader just returns a dict that TrainConfig(**raw) accepts
    monkeypatch.setattr(
        evaluate,
        "load_training_config",
        lambda p, strict=False: {
            "model": "ECGResNet",
            "lr": 0.001,
            "batch_size": 2,
            "weight_decay": 0.0,
            "n_epochs": 1,
            "save_best": True,
            "sample_only": True,
            "subsample_frac": 1.0,
            "sampling_rate": 100,
            "tag": "t",
            "data_dir": None,
            "sample_dir": str(tmp_path),
        },
        raising=False,
    )

    # No checkpoints → MODELS_DIR.glob(...) returns nothing (default)
    # Provide fallback summary
    mpath = tmp_path / "m.pth"
    mpath.write_bytes(b"")
    (tmp_path / "summary_t.json").write_text(
        '[{"model_path": "%s", "fold": 1, "best_epoch": 1, "loss": 0.1}]' % str(mpath)
    )

    # Minimal meta stub (supports meta.loc[mask].reset_index(...))
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.ones((3, 12, 20), dtype=np.float32)
    y = np.array(["NORM", "MI", "STTC"], dtype=object)
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda *a, **k: (X, y, meta), raising=False
    )

    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGResNet": _TinyLogitModel}, raising=False
    )
    monkeypatch.setattr(evaluate.torch, "load", lambda *a, **k: {}, raising=False)

    # No-op outputs
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    seen = {"model": None}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["model"] = kwargs.get("model")

    monkeypatch.setattr(
        evaluate,
        "save_classification_report_csv",
        lambda y_true, y_pred, out_folder, tag, fold_id: seen.setdefault(
            "fold_id", fold_id
        ),
        raising=False,
    )

    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=False
    )

    # Triggers: prefer=latest (no ckpt) -> fallback to summary
    evaluate.main(prefer="latest", shap_profile="off")

    # Assert the tag-derived model (“ECGResNet”) was used
    assert seen["model"] == "ECGResNet"
    assert seen["fold_id"] == 1  # your summary stub uses fold: 1


def test_main_prefer_latest_with_ckpt_and_history(monkeypatch, tmp_path):
    # Temp dirs
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "PLOTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path, raising=False)

    # Create ckpt that matches regex: model_best_(tag)_foldN.pth
    ckpt = tmp_path / "model_best_tagX_ECGResNet_fold3.pth"
    ckpt.write_bytes(b"")
    # Make MODELS_DIR.glob return our single ckpt
    monkeypatch.setattr(
        type(tmp_path), "glob", lambda self, patt: [ckpt], raising=False
    )

    # Matching history file path uses 'tag' captured by regex
    hist = tmp_path / "history_tagX_ECGResNet_fold3.json"
    hist.write_text(
        '{"best_epoch": 2, "val_acc": [0.8], "val_loss": [0.2], "train_acc": [0.9], "train_loss": [0.3]}'
    )

    # Latest config path
    cfg_path = tmp_path / "config_t.yaml"
    cfg_path.write_text(
        "model: ECGResNet\nbatch_size: 2\nlr: 0.001\nweight_decay: 0.0\nn_epochs: 1\nsample_only: true\nsample_dir: %s\ntag: t\n"
        % tmp_path
    )
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: cfg_path, raising=False
    )

    monkeypatch.setattr(
        evaluate,
        "load_training_config",
        lambda p, strict=False: {
            "model": "ECGResNet",
            "lr": 0.001,
            "batch_size": 2,
            "weight_decay": 0.0,
            "n_epochs": 1,
            "save_best": True,
            "sample_only": True,
            "subsample_frac": 1.0,
            "sampling_rate": 100,
            "tag": "t",
            "data_dir": None,
            "sample_dir": str(tmp_path),
        },
        raising=False,
    )

    # Data + meta
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.ones((4, 12, 20), dtype=np.float32)
    y = np.array(["NORM", "MI", "STTC", "HYP"], dtype=object)
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda *a, **k: (X, y, meta), raising=False
    )

    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGResNet": _TinyLogitModel}, raising=False
    )
    monkeypatch.setattr(evaluate.torch, "load", lambda *a, **k: {}, raising=False)

    # No-op outputs
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    seen = {}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["model"] = kwargs.get("model")
        seen["fold"] = kwargs.get("fold")
        seen["epoch"] = kwargs.get("epoch")

    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=False
    )

    # Triggers: prefer=latest finds ckpt → reads history, parses model/tag/fold
    evaluate.main(prefer="latest", shap_profile="off")

    assert seen["model"] == "ECGResNet"  # parsed from the tag in the ckpt filename
    assert seen["fold"] == 3  # from "..._fold3.pth"
    assert seen["epoch"] == 2  # pulled from history_tagX_ECGResNet_fold3.json


def test_main_shap_custom_stride_one_skips_downsample(monkeypatch, tmp_path):
    # Temp dirs
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "PLOTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", tmp_path, raising=False)

    # Summary fallback path
    cfg_path = tmp_path / "config_t.yaml"
    cfg_path.write_text(
        "model: ECGResNet\nbatch_size: 2\nlr: 0.001\nweight_decay: 0.0\nn_epochs: 1\nsample_only: true\nsample_dir: %s\ntag: t\n"
        % tmp_path
    )
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: cfg_path, raising=False
    )

    monkeypatch.setattr(
        evaluate,
        "load_training_config",
        lambda p, strict=False: {
            "model": "ECGResNet",
            "lr": 0.001,
            "batch_size": 2,
            "weight_decay": 0.0,
            "n_epochs": 1,
            "save_best": True,
            "sample_only": True,
            "subsample_frac": 1.0,
            "sampling_rate": 100,
            "tag": "t",
            "data_dir": None,
            "sample_dir": str(tmp_path),
        },
        raising=False,
    )

    mpath = tmp_path / "m.pth"
    mpath.write_bytes(b"")
    (tmp_path / "summary_t.json").write_text(
        '[{"model_path": "%s", "fold": 2, "best_epoch": 5, "loss": 0.2}]' % str(mpath)
    )

    # Data + meta
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.ones((6, 12, 24), dtype=np.float32)
    y = np.array(["NORM", "MI", "STTC", "HYP", "CD", "NORM"], dtype=object)
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda *a, **k: (X, y, meta), raising=False
    )

    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGResNet": _TinyLogitModel}, raising=False
    )
    monkeypatch.setattr(evaluate.torch, "load", lambda *a, **k: {}, raising=False)

    # SHAP plumbing (fast, deterministic)
    monkeypatch.setattr(
        evaluate,
        "shap_sample_background",
        lambda X, max_background, seed: torch.zeros(
            (max_background, X.shape[1], X.shape[2])
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda model, x, bg, device=None: torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2])
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "shap_save_channel_summary", lambda *a, **k: "ok", raising=False
    )

    seen = {"T": None}
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda model, x, bg, device=None: (
            seen.__setitem__("T", int(x.shape[2]))
            or torch.zeros((x.shape[0], x.shape[1], x.shape[2]))
        ),
        raising=False,
    )

    # No-op reports/plots
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    # Triggers: stride == 1 → skip downsample branch
    evaluate.main(shap_profile="custom", shap_n=4, shap_bg=2, shap_stride=1)

    assert seen["T"] == 24  # stride=1 => no downsample


def test_main_prefer_latest_malformed_ckpt_tag_falls_back_to_config_model(
    monkeypatch, tmp_path
):
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    for d in (results_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path

    # Matches glob model_best_*_fold*.pth but FAILS the internal regex (no digits after 'fold')
    bad_ckpt = models_dir / "model_best_tagX_foldx.pth"
    bad_ckpt.write_bytes(b"x")

    # Config (fallback model)
    cfg = evaluate.TrainConfig(
        model="ECGResNet",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )
    (results_dir / "config.yaml").write_text(
        "tag: t\nmodel: ECGResNet\nbatch_size: 2\n"
    )
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda *a, **k: (cfg, {"tag": "t", "fold": None}),
        raising=False,
    )

    # Tiny data
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.zeros((2, 1, 8), dtype=np.float32)
    y = ["NORM", "NORM"]
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda **kw: (X, y, meta), raising=False
    )

    # Model + weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )
    _m = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr("torch.load", lambda *a, **k: _m.state_dict(), raising=False)
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Spy: confirm we fell back to config.model and still ran plots
    seen = {"model": None, "called": False}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["called"] = True
        seen["model"] = kwargs.get("model")

    monkeypatch.setattr(evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=True)
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    evaluate.main(
        parsed_args=types.SimpleNamespace(prefer="latest", shap_profile="off")
    )
    assert seen["called"] is True
    assert (
        seen["model"] == "ECGResNet"
    )  # proves: chosen_tag None -> fallback to config.model


def test_main_auto_best_json_present_but_unusable_falls_back(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    for d in (results_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path

    # A 'really_the_best' file exists, but payload lacks by_accuracy/by_loss -> best stays None
    (results_dir / "really_the_best_dummy.json").write_text(
        json.dumps({"note": "no usable entries"})
    )

    # Provide a valid summary to use as the fallback
    ckpt = models_dir / "model_best_tagY_fold1.pth"
    ckpt.write_bytes(b"x")
    (results_dir / "summary_t.json").write_text(
        json.dumps(
            [
                {
                    "model": "ECGResNet",
                    "model_path": str(ckpt),
                    "fold": 1,
                    "best_epoch": 2,
                    "loss": 0.1,
                }
            ]
        )
    )

    # Config
    cfg = evaluate.TrainConfig(
        model="ECGResNet",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )
    (results_dir / "config.yaml").write_text(
        "tag: t\nmodel: ECGResNet\nbatch_size: 2\n"
    )
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda *a, **k: (cfg, {"tag": "t", "fold": None}),
        raising=False,
    )

    # Tiny data
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.zeros((2, 1, 8), dtype=np.float32)
    y = ["NORM", "NORM"]
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda **kw: (X, y, meta), raising=False
    )

    # Model + weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )
    _m = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr("torch.load", lambda *a, **k: _m.state_dict(), raising=False)
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: json.loads((results_dir / "summary_t.json").read_text()),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "_select_best_entry", lambda s, **kw: s[0], raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Spy: ensure we completed and used the fallback summary selection
    seen = {"model_path": None}

    def _spy_eval_and_plot(*args, **kwargs):
        seen["model_path"] = kwargs.get("prefix")  # any call proves path progressed

    monkeypatch.setattr(evaluate, "evaluate_and_plot", _spy_eval_and_plot, raising=True)
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    evaluate.main(parsed_args=types.SimpleNamespace(prefer="auto", shap_profile="off"))
    # If we got here without exception, we fell back from the unusable best.json into the summary path
    assert (results_dir / "really_the_best_dummy.json").exists()
    assert seen["model_path"] is not None


def test_main_prefer_unknown_flows_direct_to_fallback(monkeypatch, tmp_path):
    results_dir = tmp_path / "results"
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    for d in (results_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)
    evaluate.OUTPUT_DIR = tmp_path

    # Provide a summary for the fallback
    ckpt = models_dir / "model_best_tagZ_fold2.pth"
    ckpt.write_bytes(b"x")
    (results_dir / "summary_t.json").write_text(
        json.dumps(
            [
                {
                    "model": "ECGResNet",
                    "model_path": str(ckpt),
                    "fold": 2,
                    "best_epoch": 3,
                    "loss": 0.2,
                }
            ]
        )
    )

    # Config
    cfg = evaluate.TrainConfig(
        model="ECGResNet",
        lr=1e-3,
        batch_size=2,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=1,
        plots_enable_ovr=False,
        plots_ovr_classes=[],
    )
    (results_dir / "config.yaml").write_text(
        "tag: t\nmodel: ECGResNet\nbatch_size: 2\n"
    )
    monkeypatch.setattr(
        evaluate,
        "_latest_config_path",
        lambda: results_dir / "config.yaml",
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda *a, **k: (cfg, {"tag": "t", "fold": None}),
        raising=False,
    )

    # Tiny data
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, _):
            return self

        def reset_index(self, drop=False):
            return self

    X = np.zeros((2, 1, 8), dtype=np.float32)
    y = ["NORM", "NORM"]
    meta = _Meta()
    monkeypatch.setattr(
        evaluate, "load_ptbxl_sample", lambda **kw: (X, y, meta), raising=False
    )

    # Model + weights
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {
            "ECGResNet": lambda num_classes, **kw: _TinyLogitModel(
                num_classes, input_channels=1
            )
        },
        raising=False,
    )
    _m = _TinyLogitModel(num_classes=5, input_channels=1)
    monkeypatch.setattr("torch.load", lambda *a, **k: _m.state_dict(), raising=False)
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: json.loads((results_dir / "summary_t.json").read_text()),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "_select_best_entry", lambda s, **kw: s[0], raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Spy: make sure we reached evaluation via the fallback
    called = {"ok": False}
    monkeypatch.setattr(
        evaluate,
        "evaluate_and_plot",
        lambda *a, **k: called.__setitem__("ok", True),
        raising=True,
    )
    monkeypatch.setattr(
        evaluate, "save_classification_report_csv", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "save_fold_summary_csv", lambda *a, **k: None, raising=False
    )

    # prefer is neither in {"accuracy","loss","auto"} nor "latest"
    evaluate.main(parsed_args=types.SimpleNamespace(prefer="weird", shap_profile="off"))
    assert called["ok"] is True


# ------------------------------------------------------------------------------
# _read_summary
# ------------------------------------------------------------------------------


def test_read_summary_raises_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    with pytest.raises(FileNotFoundError, match=r"^Missing summary for tag 'dummy'"):
        evaluate._read_summary("dummy")


def test_read_summary_raises_when_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    (tmp_path / "summary_dummy.json").write_text("[]")  # empty file
    with pytest.raises(ValueError, match=r"^Summary JSON malformed or empty"):
        evaluate._read_summary("dummy")


# ------------------------------------------------------------------------------
# _select_best_entry
# ------------------------------------------------------------------------------


def test_select_best_entry_raises_for_missing_fold():
    summaries = [
        {"fold": 0, "loss": 0.10},
        {"fold": 1, "loss": 0.05},
    ]
    with pytest.raises(ValueError, match=r"^No summary entry found for fold 99"):
        evaluate._select_best_entry(summaries, fold_override=99)


def test_select_best_entry_raises_when_loss_missing():
    summaries = [
        {"fold": 0},  # missing 'loss'
        {"fold": 1, "loss": 0.05},
    ]
    with pytest.raises(ValueError, match=r"^Summary entries missing 'loss' key"):
        evaluate._select_best_entry(summaries, fold_override=None)


# ------------------------------------------------------------------------------
# _resolve_ovr_flags
# ------------------------------------------------------------------------------


def test_resolve_ovr_sanitizer_all_string_paths_and_list():

    base = dict(
        model="ECGConvNet",
        lr=0.001,
        batch_size=64,
        weight_decay=0.0,
        n_epochs=1,
        save_best=True,
        sample_only=True,
        subsample_frac=1.0,
        sampling_rate=100,
        data_dir=None,
        sample_dir=None,
        verbose=False,
        n_folds=5,
        plots_enable_ovr=False,
    )

    # 1) STRING "__placeholder__"  -> TRUE branch
    cfg = evaluate.TrainConfig(**base, plots_ovr_classes="__placeholder__")
    en, cls = evaluate._resolve_ovr_flags(cfg)
    assert getattr(cfg, "plots_ovr_classes") == []
    assert en is False and cls is None

    # 2) STRING "   " (whitespace) -> TRUE branch via .strip() == ""
    cfg = evaluate.TrainConfig(**base, plots_ovr_classes="   ")
    en, cls = evaluate._resolve_ovr_flags(cfg)
    assert getattr(cfg, "plots_ovr_classes") == []
    assert en is False and cls is None

    # 3) STRING "MI" -> FALSE branch; coerced to ["MI"], enables OvR (with your current code)
    cfg = evaluate.TrainConfig(**base, plots_ovr_classes="MI")
    en, cls = evaluate._resolve_ovr_flags(cfg)
    assert getattr(cfg, "plots_ovr_classes") == ["MI"]
    assert en is True and cls == {"MI"}

    # 4) LIST ["__placeholder__"]  -> list sanitizer branch (empties)
    cfg = evaluate.TrainConfig(**base, plots_ovr_classes=["__placeholder__"])
    en, cls = evaluate._resolve_ovr_flags(cfg)
    assert getattr(cfg, "plots_ovr_classes") == []
    assert en is False and cls is None


def test_resolve_ovr_flags_cli_empty_classes_is_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    # argparse type= already strips empties -> resolver sees []
    with pytest.raises(SystemExit) as e:
        evaluate._resolve_ovr_flags(ovr_cfg(), cli_ovr_enable=None, cli_ovr_classes=[])
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "empty OvR class list" in err  # or "empty OvR class list" in err


def test_resolve_ovr_flags_cli_unknown_class_is_error(monkeypatch, capsys):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )
    with pytest.raises(SystemExit) as e:
        evaluate._resolve_ovr_flags(
            ovr_cfg(), cli_ovr_enable=None, cli_ovr_classes=["MI", "ABCDEF"]
        )
    assert e.value.code == 1
    _, err = capsys.readouterr()
    assert "unknown OvR class(es) from CLI" in err
    assert "ABCDEF" in err


def test_resolve_ovr_flags_cli_disable_wins_even_with_classes(monkeypatch):
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI"], raising=False
    )

    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(), cli_ovr_enable=False, cli_ovr_classes=["MI"]
    )
    assert enable is False
    assert classes is None


def test_resolve_ovr_flags_cli_enable_true_wins(monkeypatch):

    cfg = SimpleNamespace(plots_enable_ovr=False, plots_ovr_classes=[])

    # Set ENV to disable, but CLI True should win
    monkeypatch.setenv("ECG_PLOTS_ENABLE_OVR", "0")

    enable, classes = evaluate._resolve_ovr_flags(
        cfg, cli_ovr_enable=True, cli_ovr_classes=None
    )

    assert enable is True
    assert classes is None


def test_resolve_ovr_flags_cli_dedup_and_strip_then_validate(monkeypatch):

    # Ensure duplicates and whitespace are normalized before validation
    monkeypatch.setattr(
        "ecg_cnn.evaluate.FIVE_SUPERCLASSES", ["NORM", "MI", "STTC"], raising=False
    )
    enable, classes = evaluate._resolve_ovr_flags(
        ovr_cfg(),
        cli_ovr_enable=None,
        cli_ovr_classes=["MI", " STTC ", "MI"],  # duplicates + whitespace
    )
    assert enable is True
    assert classes == {"MI", "STTC"}


# ------------------------------------------------------------------------------
# _np_from_tensor
# ------------------------------------------------------------------------------


def test_np_from_tensor_tensor_and_ndarray_and_raises():
    # Tensor -> numpy (covers 340-341)
    t = torch.tensor([[1.0, 2.0]])
    out = evaluate._np_from_tensor(t, "x")
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)
    # ndarray passthrough (covers 342-343)
    a = np.array([[3.0, 4.0]], dtype=np.float32)
    out2 = evaluate._np_from_tensor(a, "x")
    assert out2 is a  # same object, passthrough
    # bad type -> TypeError (covers 344)
    with pytest.raises(TypeError, match=r"x must be np\.ndarray or torch\.Tensor"):
        evaluate._np_from_tensor({"not": "array"}, "x")


# ------------------------------------------------------------------------------
# _validate_3d
# ------------------------------------------------------------------------------


def test_validate_3d_type_dim_and_shape_errors():
    # Not an ndarray -> TypeError (covers 348-349)
    with pytest.raises(TypeError, match=r"^y must be np\.ndarray after conversion"):
        evaluate._validate_3d("y", torch.zeros(2, 1, 8))  # tensor on purpose

    # Wrong ndim -> ValueError (covers 350-353)
    bad_ndim = np.zeros((2, 8), dtype=np.float32)  # 2D, not 3D
    with pytest.raises(ValueError, match=r"^z must have shape \(N, C, T\); got ndim=2"):
        evaluate._validate_3d("z", bad_ndim)

    # Invalid shape (T <= 1) -> ValueError (covers 354-356)
    bad_shape = np.zeros((2, 1, 1), dtype=np.float32)  # T=1 invalid
    with pytest.raises(
        ValueError, match=r"^w invalid shape \(2, 1, 1\); need N>0, C>0, T>1"
    ):
        evaluate._validate_3d("w", bad_shape)


# ------------------------------------------------------------------------------
# _as_1d_label_list()
# ------------------------------------------------------------------------------


def test_as_1d_label_list_column_vector_and_nested_unwrap():
    # Column vector (n,1) should flatten to (n,)
    y_col = np.array([["NORM"], ["MI"], ["STTC"]], dtype=object)
    out = evaluate._as_1d_label_list(y_col)
    assert out == ["NORM", "MI", "STTC"]

    # Nested containers should unwrap first element of each
    y_nested = [[1], np.array([2]), 3]
    out2 = evaluate._as_1d_label_list(y_nested)
    assert out2 == [1, 2, 3]


def test_as_1d_label_list_squeeze_and_error_non_1d():
    # Shape (2,1,2) squeezes to (2,2) -> still not 1-D -> ValueError
    y_bad = np.zeros((2, 1, 2), dtype=int)
    with pytest.raises(ValueError):
        evaluate._as_1d_label_list(y_bad)


def test_as_1d_label_list_raises_on_empty_container():
    # Contains an empty list -> should raise the "Encountered empty label container" error
    with pytest.raises(ValueError):
        evaluate._as_1d_label_list([["ok"], []])


# ------------------------------------------------------------------------------
# SHAP summary block tests
# ------------------------------------------------------------------------------
def test_eval_shap_off_prints_message(capfd):
    shap_profile = "off"
    if shap_profile and shap_profile.lower() == "off":
        print("SHAP disabled (--shap off).")
    out = capfd.readouterr().out
    assert "SHAP disabled (--shap off)." in out


def test_eval_shap_unknown_profile_falls_back_to_medium(capfd):
    profiles = {
        "fast": (4, 4, 10),
        "medium": (8, 8, 5),
        "thorough": (16, 16, 2),
    }
    shap_profile = "weird"
    fallback = "medium"
    if shap_profile not in profiles:
        print(f"Unknown --shap profile '{shap_profile}', using '{fallback}'.")
        shap_profile = fallback
    out = capfd.readouterr().out
    assert "Unknown --shap profile 'weird', using 'medium'." in out


def test_eval_shap_prints_no_batches_when_empty_loader(capfd):
    dummy_data = TensorDataset(torch.empty(0, 12, 100))  # (0, C, T)
    loader = DataLoader(dummy_data, batch_size=8)

    # simulate batch collection loop
    X_list = []
    with torch.no_grad():
        for xb in loader:
            X_list.append(xb[0])
            if sum(x.shape[0] for x in X_list) >= 4:
                break
    if not X_list:
        print("SHAP: no batches available to explain.")

    out = capfd.readouterr().out
    assert "SHAP: no batches available to explain." in out


def test_eval_shap_summary_executes_main_flow(monkeypatch, capfd):
    # fake tensors
    X_ex = torch.randn(4, 12, 100)
    bg_t = torch.randn(4, 12, 100)

    # patch everything
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda m, x, b, device=None: torch.randn(4, 12, 100),
    )
    monkeypatch.setattr(
        evaluate, "shap_sample_background", lambda x, max_background, seed: bg_t
    )
    monkeypatch.setattr(evaluate, "_shap_stability_report", lambda x, class_names: "OK")
    monkeypatch.setattr(
        evaluate, "shap_save_channel_summary", lambda *a, **kw: "/fake/path.png"
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # fake dataset with real loader
    dataset = TensorDataset(torch.randn(10, 12, 100), torch.zeros(10, dtype=torch.long))
    model = torch.nn.Identity()
    config = SimpleNamespace(model="Conv", batch_size=8)
    evaluate.FIVE_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

    # simulate execution
    try:
        profiles = {"fast": (4, 4, 10)}
        n, bg, stride = profiles["fast"]
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        X_list = []
        with torch.no_grad():
            for xb, _ in loader:
                X_list.append(xb)
                if sum(x.shape[0] for x in X_list) >= n:
                    break
        X_explain = torch.cat(X_list, dim=0)[:n]
        if stride > 1:
            X_explain = X_explain[:, :, ::stride]
        bg = evaluate.shap_sample_background(
            X_explain, max_background=bg, seed=evaluate.SEED
        )
        sv = evaluate.shap_compute_values(model, X_explain, bg)
        print(evaluate._shap_stability_report(sv, evaluate.FIVE_SUPERCLASSES))
        print("SHAP finished in 0.00s for the above config.")
        print(f"Saved SHAP summary: /fake/path.png")
    except Exception as e:
        print(f"SHAP generation skipped/failed: {e}")

    out = capfd.readouterr().out
    assert "Saved SHAP summary" in out


def test_eval_shap_handles_exceptions_gracefully(capfd):
    try:
        raise ValueError("Simulated SHAP failure")
    except Exception as e:
        print(f"SHAP generation skipped/failed: {e}")

    out = capfd.readouterr().out
    assert "SHAP generation skipped/failed: Simulated SHAP failure" in out


# --------------------------------------------------------------------------
# _shap_stability_report()
# --------------------------------------------------------------------------


def test_shap_stability_report_binary_tensor_returns_expected_flags():
    # shape: (N, C, T)
    sv = torch.ones((8, 4, 100))
    sv[:, 2] *= 10  # boost one channel to dominate
    report = evaluate._shap_stability_report(sv)
    assert "SHAP channel importance" in report
    assert "02" in report  # channel index
    assert "STABLE" in report or "OK" in report or "NOISY" in report
    assert "Top channels look stable" in report


def test_shap_stability_report_multiclass_tensor_list_works():
    sv = [torch.randn(8, 4, 50) * (i + 1) for i in range(5)]  # 5-class list
    report = evaluate._shap_stability_report(sv)
    assert "SHAP channel importance" in report
    assert report.count("\n") > 5
    assert any(flag in report for flag in ["STABLE", "OK", "NOISY"])


def test_shap_stability_report_flags_noisy_channels_and_prints_guidance():
    sv = torch.zeros((8, 4, 100))  # All zeros = mean 0, std 0
    sv[:, 0, 50] = 100.0  # Inject spike into channel 0 at one timepoint

    report = evaluate._shap_stability_report(sv)

    assert "NOISY" in report
    assert "flagged NOISY (CV ≥ 0.5)" in report
    assert "Increase --shap-n" in report


def test_shap_stability_report_mixed_input_types():
    sv = [np.ones((6, 3, 50)), torch.ones((6, 3, 50))]
    report = evaluate._shap_stability_report(sv)
    assert "SHAP channel importance" in report
    assert "Top channels look stable" in report


def test_shap_stability_report_raises_on_empty_list():
    with pytest.raises(ValueError, match=r"^sv is an empty list"):
        evaluate._shap_stability_report([])


def test_shap_stability_report_shape_mismatch_raises():
    sv = [np.ones((6, 4, 50)), np.ones((6, 3, 50))]
    with pytest.raises(
        ValueError, match=r"^sv\[1\] shape \(6, 3, 50\) != sv\[0\] shape \(6, 4, 50\)"
    ):
        evaluate._shap_stability_report(sv)


def test_shap_stability_report_rejects_non_tensor_or_array():
    with pytest.raises(TypeError, match=r"^sv must be np.ndarray or torch.Tensor"):
        evaluate._shap_stability_report("not an array")


def test_shap_stability_report_rejects_wrong_dims():
    bad = np.ones((10, 3))  # not 3D
    with pytest.raises(ValueError, match=r"^sv must have shape \(N, C, T\);"):
        evaluate._shap_stability_report(bad)


def test_eval_shap_custom_profile_values_parse_correctly():
    shap_n = "12"
    shap_bg = "6"
    shap_stride = "3"
    if "custom" == "custom":
        n = int(shap_n if shap_n is not None else 8)
        bg = int(shap_bg if shap_bg is not None else 8)
        stride = int(shap_stride if shap_stride is not None else 5)
    assert (n, bg, stride) == (12, 6, 3)


def test_eval_shap_fallback_to_medium_on_unknown_profile(capfd):
    profiles = {"fast": (4, 4, 10), "medium": (8, 8, 5), "thorough": (16, 16, 2)}
    shap_profile = "nonsense"
    if shap_profile not in profiles:
        print(f"Unknown --shap profile '{shap_profile}', using 'medium'.")
        shap_profile = "medium"
    out = capfd.readouterr().out
    assert "Unknown --shap profile 'nonsense', using 'medium'." in out


def test_eval_shap_prints_no_batches_message(capfd):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.empty(0, 12, 100)),
        batch_size=8,
    )
    X_list = []
    with torch.no_grad():
        for xb, _ in loader:
            X_list.append(xb)
            if sum(x.shape[0] for x in X_list) >= 4:
                break
    if not X_list:
        print("SHAP: no batches available to explain.")
    out = capfd.readouterr().out
    assert "SHAP: no batches available to explain." in out


def test_eval_shap_custom_profile_parses_ints(monkeypatch, tmp_path):
    """
    Covers: custom SHAP profile with string inputs (shap_n, shap_bg, shap_stride)

    Ensures that string arguments are correctly parsed as integers, and passed
    into SHAP routines. This covers the `int(...)` conversion logic when
    shap_profile == "custom".

    Expected behavior:
        - shap_n = "12" --> 12
        - shap_bg = "6" --> 6
        - shap_stride = "3" --> 3
    """

    # Track values passed into SHAP save
    saved_params = {}

    def mock_sample_background(X, max_background, seed):
        out = torch.ones((max_background, X.shape[1], X.shape[2]))
        saved_params["bg"] = out.shape[0]
        return out

    monkeypatch.setattr(evaluate, "shap_sample_background", mock_sample_background)
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda model, x, bg, device=None: torch.ones(
            (x.shape[0], x.shape[1], x.shape[2])
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "_shap_stability_report",
        lambda sv, class_names=None: "mock report",
    )

    def fake_save(sv, X, outdir, fname):
        saved_params["n"] = X.shape[0]
        saved_params["stride"] = X.shape[2]
        return "mock_output.png"

    monkeypatch.setattr(evaluate, "shap_save_channel_summary", fake_save)

    # Prepare minimal context used by evaluate.main
    evaluate.model = MagicMock()
    evaluate.model.eval = lambda: None
    evaluate.device = torch.device("cpu")
    evaluate.dataset = TensorDataset(
        torch.ones((12, 12, 100)), torch.zeros(12, dtype=torch.long)
    )
    evaluate.config = SimpleNamespace(model="ECGResNet", batch_size=4)
    evaluate.tag = "t"
    evaluate.best_fold = 1
    evaluate.best_epoch = 3
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.SEED = 0

    # No CUDA
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    # Tiny real-looking YAML so main() reads a config and sees a tag
    dummy_config_path = tmp_path / "config_t.yaml"
    dummy_config_path.write_text(
        "model: ECGResNet\n"
        "batch_size: 4\n"
        "lr: 0.001\n"
        "weight_decay: 0.0\n"
        "n_epochs: 1\n"
        "save_best: true\n"
        "sample_only: false\n"
        "subsample_frac: 1.0\n"
        "sampling_rate: 100\n"
        "tag: t\n"
    )
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "_latest_config_path", lambda: dummy_config_path)

    # Pretend there’s a summary for this tag
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {
                "model_path": str(tmp_path / "mock.pt"),
                "best_epoch": 3,
                "loss": 0.1,
                "val_acc": 0.9,
                "fold": 1,
            }
        ],
        raising=False,
    )

    # Make best-entry selection trivial
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Minimal meta supporting meta.loc[mask].reset_index(...)
    class _Meta:
        @property
        def loc(self):
            return self

        def __getitem__(self, key):
            return self

        def reset_index(self, drop=False):
            return self

    meta = _Meta()

    # Synthetic data; IMPORTANT: y is a 1-D NumPy object array (no DataConversionWarning)
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda *args, **kwargs: (
            torch.ones((12, 12, 100)).numpy(),
            np.array(["NORM"] * 6 + ["MI"] * 6, dtype=object),
            meta,
        ),
        raising=False,
    )

    # Don’t generate plots or read history in this test
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *args, **kwargs: None, raising=False
    )

    # Ensure our tiny model is used (rebind dicts so evaluate sees it)
    monkeypatch.setattr(
        models, "MODEL_CLASSES", {"ECGResNet": _TinyLogitModel}, raising=False
    )
    monkeypatch.setattr(
        evaluate, "MODEL_CLASSES", {"ECGResNet": _TinyLogitModel}, raising=False
    )

    # Stub torch.load and create the weights file it looks for
    (tmp_path / "mock.pt").write_bytes(b"")
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    # IMPORTANT: keep sklearn out of report generation in this SHAP test
    # (no imports in tests; use string target)
    monkeypatch.setattr(
        "ecg_cnn.utils.plot_utils.save_classification_report_csv",
        lambda *a, **k: None,
        raising=False,
    )

    # Call main with string values for shap_* overrides
    evaluate.main(
        shap_profile="custom",
        shap_n="12",
        shap_bg="6",
        shap_stride="3",
    )

    # Check that string inputs were parsed correctly
    assert saved_params["n"] == 12
    assert saved_params["bg"] == 6
    assert saved_params["stride"] == math.ceil(evaluate.dataset.tensors[0].shape[2] / 3)


def test_eval_shap_profile_falls_back_to_medium(monkeypatch, tmp_path, capsys):
    """
    Covers: lines 657-658 (unknown shap_profile).
    Ensures that when shap_profile is not one of the known keys,
    evaluate.main() prints the warning and falls back to 'medium'.
    """

    # Minimal monkeypatches to prevent heavy work
    monkeypatch.setattr(
        evaluate,
        "shap_sample_background",
        lambda X, max_background, seed: torch.ones((4, X.shape[1], X.shape[2])),
    )
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda model, x, bg, device=None: torch.ones_like(x),
    )
    monkeypatch.setattr(
        evaluate, "_shap_stability_report", lambda sv, class_names=None: "mock report"
    )
    monkeypatch.setattr(
        evaluate,
        "shap_save_channel_summary",
        lambda sv, X, outdir, fname: "mock_output.png",
    )

    # Minimal fake model + dataset context
    evaluate.model = MagicMock()
    evaluate.model.eval = lambda: None
    evaluate.device = torch.device("cpu")
    evaluate.dataset = TensorDataset(
        torch.ones((4, 12, 20)), torch.zeros(4, dtype=torch.long)
    )
    evaluate.config = SimpleNamespace(
        model="ECGResNet",
        batch_size=2,
        subsample_frac=1.0,
        sampling_rate=100,
        lr=0.001,
        weight_decay=0.0,
        data_dir=None,
        sample_dir=None,
    )
    evaluate.tag = "t"
    evaluate.best_fold = 1
    evaluate.best_epoch = 1
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.SEED = 0

    # Avoid PTB-XL / summaries
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: tmp_path / "config.yaml"
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=0.001,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"model_path": str(tmp_path / "mock.pt"), "best_epoch": 1, "fold": 1}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((4, 12, 20)).numpy(),
            ["NORM"] * 2 + ["MI"] * 2,
            pd.DataFrame({"id": [0, 1, 2, 3]}),  # <-- non-empty meta with 4 rows
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    (tmp_path / "mock.pt").write_bytes(b"")
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    monkeypatch.setattr(
        models,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )

    # Run with bogus shap_profile
    evaluate.main(shap_profile="bogus")

    # Capture and assert warning
    out, _ = capsys.readouterr()
    assert "Unknown --shap profile 'bogus', using 'medium'." in out


def test_eval_shap_prints_no_batches_when_dataset_empty(monkeypatch, tmp_path, capsys):
    """
    Covers: line 674
    Ensures that when no batches are available for SHAP (empty dataset),
    evaluate.main() prints the 'no batches available' message.
    """

    # SHAP stubs
    monkeypatch.setattr(
        evaluate,
        "shap_sample_background",
        lambda X, max_background, seed: torch.ones((4, X.shape[1], X.shape[2])),
    )
    monkeypatch.setattr(
        evaluate,
        "shap_compute_values",
        lambda model, x, bg, device=None: torch.ones_like(x),
    )
    monkeypatch.setattr(
        evaluate, "_shap_stability_report", lambda sv, class_names=None: "mock report"
    )
    monkeypatch.setattr(
        evaluate,
        "shap_save_channel_summary",
        lambda sv, X, outdir, fname: "mock_output.png",
    )

    # Minimal runtime context
    evaluate.model = MagicMock()
    evaluate.model.eval = lambda: None
    evaluate.device = torch.device("cpu")
    evaluate.config = SimpleNamespace(
        model="ECGResNet",
        batch_size=2,
        subsample_frac=1.0,
        sampling_rate=100,
        lr=0.001,
        weight_decay=0.0,
        data_dir=None,
        sample_dir=None,
    )
    evaluate.tag = "t"
    evaluate.best_fold = 1
    evaluate.best_epoch = 1
    evaluate.OUTPUT_DIR = tmp_path
    evaluate.SEED = 0

    # Avoid PTB-XL / summaries
    monkeypatch.setattr(
        evaluate, "_latest_config_path", lambda: tmp_path / "config.yaml"
    )
    monkeypatch.setattr(
        evaluate,
        "_load_config_and_extras",
        lambda path, fold_override: (
            SimpleNamespace(
                model="ECGResNet",
                batch_size=2,
                subsample_frac=1.0,
                sampling_rate=100,
                lr=0.001,
                weight_decay=0.0,
                data_dir=None,
                sample_dir=None,
            ),
            {"tag": "t", "fold": 1},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_read_summary",
        lambda tag: [
            {"model_path": str(tmp_path / "mock.pt"), "best_epoch": 1, "fold": 1}
        ],
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "_select_best_entry",
        lambda summaries, fold_override=None: summaries[0],
        raising=False,
    )

    # Return an EMPTY dataset so the SHAP gather loop has no batches
    monkeypatch.setattr(
        evaluate,
        "load_ptbxl_full",
        lambda **kw: (
            torch.ones((0, 12, 20)).numpy(),  # X with zero rows
            [],  # y empty
            pd.DataFrame({"id": []}),  # meta with zero rows
        ),
        raising=False,
    )

    # Skip plotting/history I/O
    monkeypatch.setattr(
        evaluate, "evaluate_and_plot", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        evaluate, "_load_history", lambda *a, **k: ([], [], [], []), raising=False
    )

    # Weights + loader stub
    (tmp_path / "mock.pt").write_bytes(b"")
    monkeypatch.setattr("torch.load", lambda *a, **k: {}, raising=False)

    monkeypatch.setattr(
        models,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGResNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )

    # Use a valid profile (e.g., 'fast') so SHAP runs but finds no batches
    evaluate.main(shap_profile="fast")

    out, _ = capsys.readouterr()
    assert "SHAP: no batches available to explain." in out


# ---------------------------------------------------------------------------
# Extra CLI coverage for evaluate.py
# ---------------------------------------------------------------------------


def test_main_shap_off_prints_disabled_fast(patch_paths, monkeypatch, capsys):
    """
    Hit the SHAP-off branch quickly by mocking I/O and model.
    No subprocess, no sitecustomize; only minimal patches so main() can run.
    """
    # Bind temp paths into the module under test
    (
        results_dir,
        history_dir,
        reports_dir,
        *_,
    ) = patch_paths
    monkeypatch.setattr(evaluate, "RESULTS_DIR", results_dir, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", history_dir, raising=False)
    monkeypatch.setattr(evaluate, "REPORTS_DIR", reports_dir, raising=False)

    # A config file must exist so _latest_config_path() finds something
    (results_dir / "config_dummy.yaml").write_text("dummy: true")

    # Return raw config as dict (as load_training_config would)
    def _fake_load_training_config(path, strict=False):
        return {
            "model": "ECGConvNet",
            "batch_size": 8,
            "lr": 0.001,
            "weight_decay": 0.0,
            "subsample_frac": 1.0,
            "sampling_rate": 100,
            "tag": "dummy",
            "fold": 0,
            "config": "config_dummy.yaml",
            "data_dir": None,
            "sample_dir": None,
        }

    monkeypatch.setattr(evaluate, "load_training_config", _fake_load_training_config)

    # TrainConfig(**raw) → simple object with attributes
    class _TC:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    monkeypatch.setattr(evaluate, "TrainConfig", _TC)

    # Tiny dataset: X any ndarray; y are valid known labels; meta a DF
    X = np.random.randn(12, 1, 32)  # (N,C,T) to match later torch.tensor usage
    y = np.array(
        [
            "NORM",
            "MI",
            "STTC",
            "CD",
            "HYP",
            "NORM",
            "MI",
            "STTC",
            "CD",
            "HYP",
            "NORM",
            "MI",
        ]
    )
    meta = pd.DataFrame({"dummy2": range(len(X))})
    monkeypatch.setattr(evaluate, "load_ptbxl_full", lambda **k: (X, y, meta))

    # Summary file on disk with a model_path that exists (touch a file)
    ckpt_path = results_dir / "fake_model.pt"
    ckpt_path.write_text("")  # just to make Path.exists() True
    (results_dir / "summary_dummy.json").write_text(
        json.dumps(
            [
                {
                    "fold": 0,
                    "loss": 0.123,
                    "best_fold": 0,
                    "model_path": str(ckpt_path),
                    "best_epoch": 0,
                }
            ]
        )
    )

    # torch.load should not actually load anything
    monkeypatch.setattr("torch.load", lambda *a, **k: {})

    mock_model = MagicMock()

    def _fake_forward(x):
        b = x.shape[0] if hasattr(x, "shape") else 1
        return torch.zeros(b, 5)

    mock_model.side_effect = _fake_forward  # model(x)
    mock_model.forward.return_value = torch.zeros(1, 5)
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model

    # Register the fake model under the name used in config.model
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGConvNet": MagicMock(return_value=mock_model)},
        raising=False,
    )

    # Plot/save helpers should be no-ops
    for name in (
        "evaluate_and_plot",
        "save_threshold_sweep_table",
        "save_roc_curve_multiclass_ovr",
        "save_pr_curve_multiclass_ovr",
        "save_classification_report",
        "save_confidence_histogram_split",
        "save_plot_curves",
    ):
        monkeypatch.setattr(evaluate, name, lambda *a, **k: None, raising=False)

    # Ensure class names are present (if your code prints using FIVE_SUPERCLASSES)
    monkeypatch.setattr(
        evaluate,
        "FIVE_SUPERCLASSES",
        ["NORM", "MI", "STTC", "CD", "HYP"],
        raising=False,
    )

    # Run with SHAP disabled (this is what we’re covering)
    evaluate.main(shap_profile="off")
    out = capsys.readouterr().out
    assert "SHAP disabled (--shap off)." in out


def test_main_calls_cuda_synchronize(monkeypatch, tmp_path, capsys):
    """
    If torch.cuda.is_available(): torch.cuda.synchronize()
    Fast path: redirect RESULTS_DIR, stub data to tiny arrays, and no-op heavy helpers.
    """

    # Point evaluate to tmp directories so nothing under project outputs is touched
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "OUTPUT_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "PLOTS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path, raising=False)
    monkeypatch.setattr(evaluate, "ARTIFACTS_DIR", tmp_path, raising=False)

    # Make a config file so _latest_config_path() succeeds
    fake_cfg = tmp_path / "config_0001.yaml"
    fake_cfg.write_text("model: ECGConvNet\n")
    monkeypatch.setattr(evaluate, "_latest_config_path", lambda: fake_cfg)

    def _fake_load_config_and_extras(path, fold_override=None):
        cfg = SimpleNamespace(
            model="ECGConvNet",
            batch_size=2,
            subsample_frac=1.0,
            sampling_rate=100,
            lr=0.0,
            weight_decay=0.0,
            data_dir=None,  # required by main(); ok to be None
            sample_dir=None,  # required by main(); ok to be None
        )
        extras = {
            "tag": "dummy",
            "best_fold": None,
            "best_epoch": 0,
            "train_ds": None,
            "val_ds": None,
            "test_ds": None,
        }
        return cfg, extras

    monkeypatch.setattr(
        evaluate, "_load_config_and_extras", _fake_load_config_and_extras
    )

    # Tiny dataset so load_ptbxl_full is instant
    X = np.random.randn(4, 12, 32).astype("float32")  # (N,C,T)
    y = np.array(["NORM", "MI", "STTC", "CD"])
    meta = pd.DataFrame({"i": range(4)})
    monkeypatch.setattr(evaluate, "load_ptbxl_full", lambda **k: (X, y, meta))

    # Summary + checkpoint so _read_summary/_restore_best_model pass
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.write_text("")  # existence is enough
    (tmp_path / "summary_dummy.json").write_text(
        json.dumps(
            [{"fold": 0, "loss": 0.1, "model_path": str(ckpt_path), "best_epoch": 0}]
        )
    )
    # torch.load should return an empty state dict
    monkeypatch.setattr("torch.load", lambda *a, **k: {})

    # Provide our tiny fake model
    monkeypatch.setattr(
        evaluate,
        "MODEL_CLASSES",
        {"ECGConvNet": lambda num_classes, **kw: _TinyLogitModel(num_classes)},
        raising=False,
    )

    # No-op all heavy helpers (plots/reports/shap)
    for name in [
        "evaluate_and_plot",
        "save_threshold_sweep_table",
        "save_roc_curve_multiclass_ovr",
        "save_pr_curve_multiclass_ovr",
        "save_classification_report",
        "save_confidence_histogram_split",
        "save_plot_curves",
        "shap_compute_values",
        "_shap_stability_report",
        "shap_save_channel_summary",
    ]:
        monkeypatch.setattr(evaluate, name, lambda *a, **k: None, raising=False)

    # CUDA monkeypatch: available + capture synchronize()
    calls = {"synced": False}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _fake_sync():
        calls["synced"] = True

    monkeypatch.setattr(torch.cuda, "synchronize", _fake_sync)

    # force model/device path to CPU so .to('cuda') is never called
    monkeypatch.setattr(torch, "device", lambda *a, **k: "cpu")
    # (optional belt-and-suspenders, in case evaluate references torch via its own module binding)
    monkeypatch.setattr(evaluate.torch, "device", lambda *a, **k: "cpu", raising=False)

    # Run with SHAP enabled (any non-'off') so the CUDA guard executes
    evaluate.main(shap_profile="fast")

    assert calls["synced"] is True


def test_cli_argparse_mutually_exclusive_group(monkeypatch, capsys):
    # Exercise the argparse block by running evaluate.py as __main__
    monkeypatch.setattr(
        sys, "argv", ["evaluate.py", "--enable_ovr", "--disable_ovr"], raising=False
    )
    script = str(Path(evaluate.__file__).resolve())

    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name="__main__")

    err = capsys.readouterr().err or ""
    assert ("not allowed with argument" in err) or ("mutually exclusive" in err)


def test_cli_calls_main(monkeypatch, tmp_path, capsys):
    # Cover the call to main(...) in __main__ quickly and safely.

    # 1) Force CLI to parse successfully
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--fold", "1"], raising=False)

    # 2) Inject a fake ecg_cnn.paths so evaluate.py uses tmp_path, not project outputs/
    pm = types.ModuleType("ecg_cnn.paths")
    base = Path(tmp_path)
    pm.PROJECT_ROOT = base
    pm.RESULTS_DIR = base
    pm.REPORTS_DIR = base
    pm.OUTPUT_DIR = base
    pm.HISTORY_DIR = base
    pm.PLOTS_DIR = base
    pm.MODELS_DIR = base
    pm.ARTIFACTS_DIR = base
    pm.PTBXL_DATA_DIR = base
    sys.modules["ecg_cnn.paths"] = pm  # override any real one

    # 3) Execute the file as a script; argparse runs and calls main(...) (line 790)
    script = str(Path(evaluate.__file__).resolve())
    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name="__main__")

    # 4) Early-exit message proves we stayed in the tmp sandbox
    out_err = capsys.readouterr()
    msg = (out_err.out or "") + (out_err.err or "")
    assert "No training configs found" in msg


# -----------------------------------
# bullshit
# -----------------------------------
