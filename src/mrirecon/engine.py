"""Training and evaluation loops."""

import csv
import json
import logging
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mrirecon.data import FastMRISlices, split_volumes
from mrirecon.fft import data_consistency
from mrirecon.metrics import all_metrics, ssim
from mrirecon.models import build_model

log = logging.getLogger("mrirecon")


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on real and imaginary parts for complex outputs, on magnitude otherwise."""
    if pred.is_complex():
        return F.l1_loss(torch.view_as_real(pred), torch.view_as_real(target.to(pred.dtype)))
    return F.l1_loss(pred, target.abs())


def _to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _list_h5(directory, limit=None):
    files = sorted(Path(directory).glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"no .h5 files in {directory}")
    return files[:limit] if limit else files


def reconstruct(model, batch, use_dc=True):
    out = model(batch["image"], batch["kspace"])
    if use_dc:
        out = data_consistency(out, batch["kspace"], batch["mask"])
    return out


def train(args):
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _list_h5(args.train_dir, args.limit)
    if args.val_dir:
        train_files, val_files = files, _list_h5(args.val_dir, args.limit)
    else:
        train_files, val_files = split_volumes(files, args.val_fraction, args.seed)
    log.info("%d training and %d validation volumes", len(train_files), len(val_files))

    data_kw = dict(mask=args.mask, accel=args.accel, crop=args.crop, skip_edge_slices=args.skip_edge_slices)
    train_loader = DataLoader(
        FastMRISlices(train_files, **data_kw),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        FastMRISlices(val_files, fixed_mask=True, **data_kw),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model_kwargs = {"complex_primal": not args.magnitude}
    model = build_model(args.model, **model_kwargs).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.lr_step, gamma=0.1)

    start_epoch, best = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best = ckpt["epoch"] + 1, ckpt["best_val_ssim"]
        log.info("resumed from %s at epoch %d", args.resume, start_epoch)

    history = out_dir / "history.csv"
    if start_epoch == 0:
        history.write_text("epoch,train_loss,val_ssim,lr,seconds\n")

    for epoch in range(start_epoch, args.epochs):
        tic = time.time()
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = _to(batch, device)
            loss = l1_loss(model(batch["image"], batch["kspace"]), batch["target"])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item()
            if step % args.log_every == 0:
                log.info("epoch %d step %d/%d loss %.5f", epoch, step, len(train_loader), running / step)
        train_loss = running / max(1, len(train_loader))

        val_ssim = validate(model, val_loader, device)
        lr = optimiser.param_groups[0]["lr"]
        scheduler.step()
        seconds = time.time() - tic
        log.info("epoch %d train_loss %.5f val_ssim %.4f (%.0fs)", epoch, train_loss, val_ssim, seconds)
        with history.open("a") as fh:
            fh.write(f"{epoch},{train_loss:.6f},{val_ssim:.6f},{lr:.2e},{seconds:.1f}\n")

        best = max(best, val_ssim)
        ckpt = {
            "model": args.model,
            "model_kwargs": model_kwargs,
            "data": data_kw,
            "state_dict": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_ssim": val_ssim,
            "best_val_ssim": best,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_ssim >= best:
            torch.save(ckpt, out_dir / "best.pt")
    return best


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = []
    for batch in loader:
        batch = _to(batch, device)
        scores += ssim(reconstruct(model, batch), batch["target"]).flatten().tolist()
    return sum(scores) / max(1, len(scores))


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = build_model(ckpt["model"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def evaluate(args):
    device = pick_device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    data_kw = {**ckpt["data"]}
    for key in ("mask", "accel", "crop"):
        if getattr(args, key, None) is not None:
            data_kw[key] = getattr(args, key)
    loader = DataLoader(
        FastMRISlices(_list_h5(args.data_dir, args.limit), fixed_mask=True, **data_kw),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    label = args.label or Path(args.checkpoint).parent.name

    rows = []
    for batch in loader:
        batch = _to(batch, device)
        outputs = {"zero-filled": batch["image"], label: reconstruct(model, batch, use_dc=not args.no_dc)}
        for method, out in outputs.items():
            scores = all_metrics(out[:, 0], batch["target"][:, 0])
            for i, (fname, s) in enumerate(zip(batch["fname"], batch["slice"].tolist())):
                rows.append(
                    {"method": method, "fname": fname, "slice": s, **{k: v[i].item() for k, v in scores.items()}}
                )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_slice.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"checkpoint": str(args.checkpoint), "data": data_kw, "data_consistency": not args.no_dc, "methods": {}}
    for method in outputs:
        subset = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {
            metric: {
                "mean": statistics.fmean(r[metric] for r in subset),
                "std": statistics.pstdev(r[metric] for r in subset),
                "median": statistics.median(r[metric] for r in subset),
            }
            for metric in ("ssim", "psnr", "nmse", "nrmse")
        } | {"n_slices": len(subset)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for method, stats in summary["methods"].items():
        log.info(
            "%-14s SSIM %.4f +/- %.4f  PSNR %.2f  NMSE %.4f  (n=%d)",
            method,
            stats["ssim"]["mean"],
            stats["ssim"]["std"],
            stats["psnr"]["mean"],
            stats["nmse"]["mean"],
            stats["n_slices"],
        )
    return summary
