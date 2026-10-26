import json

import torch

from mrirecon.cli import main, run
from mrirecon.engine import l1_loss, load_checkpoint


def test_main_returns_none_for_console_script(fastmri_dir, tmp_path):
    # console_scripts wrappers call sys.exit(main()); anything but None would become a failing exit code.
    args = ["train", "--train-dir", str(fastmri_dir), "--epochs", "1", "--crop", "16", "--workers", "0"]
    assert main([*args, "--model", "fourier-pd", "--device", "cpu", "--out", str(tmp_path / "r")]) is None


def test_complex_l1():
    a = torch.tensor([1 + 1j, 0j])
    b = torch.tensor([0j, 0j])
    assert l1_loss(a, b).item() == 0.5  # mean over the four real numbers
    assert l1_loss(torch.tensor([1.0]), torch.tensor([3 + 4j])).item() == 4.0


def test_train_then_evaluate(fastmri_dir, tmp_path):
    run_dir = tmp_path / "run"
    common = ["--mask", "varden1d", "--accel", "4", "--crop", "32", "--workers", "0", "--device", "cpu"]
    best = run(
        [
            "train",
            "--model",
            "fourier-pd",
            "--train-dir",
            str(fastmri_dir),
            "--val-fraction",
            "0.25",
            "--epochs",
            "2",
            "--lr-step",
            "1",
            "--out",
            str(run_dir),
            *common,
        ]
    )
    assert 0 < best <= 1
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    assert len((run_dir / "history.csv").read_text().splitlines()) == 3

    model, ckpt = load_checkpoint(run_dir / "last.pt")
    assert ckpt["epoch"] == 1 and model.complex_primal

    # Resuming from the last epoch is a no-op that must not crash.
    main(
        [
            "train",
            "--model",
            "fourier-pd",
            "--train-dir",
            str(fastmri_dir),
            "--epochs",
            "2",
            "--resume",
            str(run_dir / "last.pt"),
            "--out",
            str(run_dir),
            *common,
        ]
    )

    results = tmp_path / "eval"
    main(
        [
            "evaluate",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data-dir",
            str(fastmri_dir),
            "--workers",
            "0",
            "--label",
            "pd",
            "--out",
            str(results),
        ]
    )
    summary = json.loads((results / "summary.json").read_text())
    assert set(summary["methods"]) == {"zero-filled", "pd"}
    assert summary["methods"]["pd"]["n_slices"] == 12
    assert summary["data"]["crop"] == 32
    rows = (results / "per_slice.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 12


def test_data_consistency_never_hurts_on_measured_lines(fastmri_dir, tmp_path):
    """With DC, a model can only change unmeasured k-space, so a fully sampled input is returned exactly."""
    run_dir = tmp_path / "run"
    common = ["--crop", "32", "--workers", "0", "--device", "cpu"]
    main(
        [
            "train",
            "--model",
            "fourier-pd",
            "--train-dir",
            str(fastmri_dir),
            "--epochs",
            "1",
            "--accel",
            "1",
            "--out",
            str(run_dir),
            *common,
        ]
    )
    summary = run(
        [
            "evaluate",
            "--checkpoint",
            str(run_dir / "last.pt"),
            "--data-dir",
            str(fastmri_dir),
            "--out",
            str(tmp_path / "e"),
            *common,
        ]
    )
    assert summary["methods"]["run"]["nrmse"]["mean"] < 1e-5
