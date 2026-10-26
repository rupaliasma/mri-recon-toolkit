import argparse
import logging

from mrirecon.masks import KINDS
from mrirecon.models import MODELS


def _data_args(p, required_defaults=True):
    d = (lambda v: v) if required_defaults else (lambda v: None)
    p.add_argument("--mask", choices=KINDS, default=d("varden1d"))
    p.add_argument("--accel", type=float, default=d(4.0), help="acceleration factor (1 / sampled fraction)")
    p.add_argument("--crop", type=int, default=d(320), help="square image-space crop")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="use only the first N volumes (smoke tests)")
    p.add_argument("--device", help="e.g. cuda:0 or cpu; defaults to cuda when available")


def build_parser():
    parser = argparse.ArgumentParser(prog="mrirecon", description="Fourier-PD / Fourier-PDUNet MRI reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model on fastMRI k-space")
    t.add_argument("--model", choices=sorted(MODELS), default="fourier-pdunet")
    t.add_argument("--magnitude", action="store_true", help="magnitude-only primal (default: complex)")
    t.add_argument("--train-dir", required=True)
    t.add_argument("--val-dir", help="if omitted, a fraction of --train-dir volumes is held out")
    t.add_argument("--val-fraction", type=float, default=0.1)
    t.add_argument("--skip-edge-slices", type=int, default=0)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-step", type=int, default=50, help="divide the learning rate by 10 every N epochs")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=200)
    t.add_argument("--resume")
    t.add_argument("--out", required=True)
    _data_args(t)

    e = sub.add_parser("evaluate", help="evaluate a checkpoint against zero-filling")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--data-dir", required=True)
    e.add_argument("--no-dc", action="store_true", help="skip the final data-consistency step")
    e.add_argument("--label", help="method name in the results (default: checkpoint folder name)")
    e.add_argument("--out", required=True)
    _data_args(e, required_defaults=False)
    return parser


def run(argv=None):
    """Parse ``argv`` and run the command, returning its result (best val SSIM or the summary)."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from mrirecon import engine

    return engine.train(args) if args.command == "train" else engine.evaluate(args)


def main(argv=None):
    run(argv)


if __name__ == "__main__":
    main()
