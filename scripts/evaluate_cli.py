# ecg_cnn/scripts/evaluate_cli.py
import argparse
import os
from ecg_cnn import evaluate as eval_mod


def main():
    p = argparse.ArgumentParser(description="Evaluate with optional OvR plot flags.")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--enable-ovr", action="store_true", help="Enable one-vs-rest PR/ROC plots."
    )
    g.add_argument(
        "--disable-ovr",
        action="store_true",
        help="Force-disable OvR plots (overrides YAML).",
    )
    p.add_argument(
        "--ovr-classes",
        nargs="*",
        default=None,
        help="Subset of class names for OvR (e.g., MI STTC). Empty means all classes.",
    )
    args, _ = p.parse_known_args()

    # Decision rules (CLI > YAML):
    # 1) --disable-ovr wins and turns it off
    # 2) else if --enable-ovr, turn it on
    # 3) else if --ovr-classes provided (even w/o --enable-ovr), turn it on
    # 4) else leave env unset -> YAML decides
    if args.disable_ovr:
        os.environ["ECG_PLOTS_ENABLE_OVR"] = "0"
    elif args.enable_ovr or (args.ovr_classes is not None):
        os.environ["ECG_PLOTS_ENABLE_OVR"] = "1"

    if args.ovr_classes is not None:
        os.environ["ECG_PLOTS_OVR_CLASSES"] = ",".join(args.ovr_classes)

    # Optional: echo effective choice (nice UX)
    eff_enable = os.environ.get("ECG_PLOTS_ENABLE_OVR", "<YAML default>")
    eff_classes = os.environ.get("ECG_PLOTS_OVR_CLASSES", "<YAML default>")
    print(f"[evaluate_cli] OvR enabled={eff_enable}, classes={eff_classes}")

    if hasattr(eval_mod, "main"):
        eval_mod.main()
    else:
        pass


if __name__ == "__main__":
    main()
