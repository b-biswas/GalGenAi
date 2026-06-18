"""
CFM Training Pipeline for COSMOS HuggingFace Dataset

Trains a Conditional Flow Matching model end-to-end on simulated galaxy
images produced by generate_fits_dataset.py. Unlike the VAE+LCFM and
VAE+CNF pipelines, CFM is a single-stage model that learns a velocity
field directly on image-space conditioned on physical galaxy properties
(magnitudes, redshift):

    v_pred = velocity_net(x_t, f, t)
    x_t    = (1 - t) * x_0 + t * x_1     (linear interpolant)
    u_t    = x_1 - x_0                   (target velocity)
    loss   = MSE(v_pred, u_t)            (weighted if ivar+mask given)

Sampling:

    f      = normalize_conditions(magnitudes, redshift, cond_stats)
    image  = cfm.sample(batch_size, device, f, num_steps=50)
    raw    = arcsinh_denorm(image, norm_stats)

Run with:
    uv run python scripts/train_cfm_cosmos.py
"""

import shutil
from pathlib import Path

from galgenai import get_device
from galgenai.config import load_config
from galgenai.data.cosmos_dataset import load_fits_dataset, make_loaders
from galgenai.data.normalization import (
    get_conditional_norm_fn,
    get_image_norm_fn,
    save_conditional_stats,
    save_image_norm_stats,
)
from galgenai.models.cfm import CFM
from galgenai.training import CFMTrainer, load_cfm_training_config


def plot_loss_history(loss_history, out_dir):
    """Plot train and val loss vs step from a trainer loss_history."""
    import matplotlib.pyplot as plt

    train_steps = [e["step"] for e in loss_history if "loss" in e]
    train_loss = [e["loss"] for e in loss_history if "loss" in e]
    val_steps = [e["step"] for e in loss_history if "val_loss" in e]
    val_loss = [e["val_loss"] for e in loss_history if "val_loss" in e]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_steps, train_loss, label="train", lw=1)
    if val_loss:
        ax.plot(val_steps, val_loss, label="val", marker="o", ms=3)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_dir) / "loss_history.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss plot to: {out_path}")


def main():
    device = get_device()
    cfg = load_config(Path("./exp/cfm_617/galgenai_config.yaml"))

    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load all required config values
    # ------------------------------------------------------------------
    try:
        # Top-level sections
        cosmos_cfg = cfg["cosmos"]
        train_cfg = cfg["training"]
        model_cfg = train_cfg["model"]
        vae_cfg = train_cfg["vae"]
        cfm_cfg = train_cfg["cfm"]
        norm_cfg = cosmos_cfg["normalization"]
        run_name = cfg["run_name"]

        # Dataset config
        dataset_path = cosmos_cfg["hf_dataset_path"]
        train_ratio = cosmos_cfg["train_ratio"]
        val_ratio = cosmos_cfg["val_ratio"]
        num_workers = cosmos_cfg["num_workers"]
        split_seed = cosmos_cfg["split_seed"]
        invert_mask = cosmos_cfg.get("invert_mask", False)

        # Training config
        output_dir = Path(train_cfg["output_dir"]) / run_name
        nx = train_cfg["nx"]
        batch_size = train_cfg["batch_size"]

        # Model config (CFM reuses in_channels / base_channels)
        in_channels = model_cfg["in_channels"]
        base_channels = model_cfg["base_channels"]

        # Image normalization mode is shared with VAE config
        image_norm_type = vae_cfg["norm_type"]

        # CFM-specific non-training params
        condition_cols = cfm_cfg["condition_cols"]

    except KeyError as e:
        raise ValueError(
            f"Missing required config value: {e}. "
            "Please ensure all required values are present in the config file."
        ) from e

    print(f"Run name: {run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy the full config file to output directory for provenance
    config_file_path = (
        Path(__file__).parent.parent
        / "src"
        / "galgenai"
        / "galgenai_config.yaml"
    )
    if config_file_path.exists():
        shutil.copy(config_file_path, output_dir / "galgenai_config.yaml")
        print(f"Copied config to: {output_dir / 'galgenai_config.yaml'}")

    cond_stats_save_path = output_dir / "cond_stats.yaml"
    norm_stats_save_path = output_dir / "norm_stats.yaml"
    condition_dim = len(condition_cols)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading FITS dataset from: {dataset_path}")

    catalog_cols = cosmos_cfg["catalog_columns"]
    mag_cols = catalog_cols["mag_cols"]
    redshift_col = catalog_cols["redshift_col"]

    dataset_raw = load_fits_dataset(
        dataset_path,
        metadata_file="metadata.csv",
        mag_cols=mag_cols,
        redshift_col=redshift_col,
        mag_sentinel=cosmos_cfg.get("mag_sentinel", 999.0),
        redshift_sentinel=cosmos_cfg.get("redshift_sentinel", -99.0),
        nx=nx,
        load_noiseless=True,  # cfm_cfg["train_on_noiseless"],
    )

    n_total = len(dataset_raw)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * (1 - train_ratio - val_ratio))
    print(
        f"Dataset sizes: {n_train} train / {n_val} val / {n_test} test "
        f"(total: {n_total})"
    )

    # ------------------------------------------------------------------
    # Normalization (image + conditioning)
    # ------------------------------------------------------------------
    print("\nLoading normalization config")
    print(f"  Image normalization: {image_norm_type}")
    image_norm_fn, image_denorm_fn, norm_stats = get_image_norm_fn(
        img_norm_type=image_norm_type,
        config=norm_cfg["image"],
        return_denorm=True,
    )
    save_image_norm_stats(norm_stats, norm_stats_save_path)
    print(f"  Image normalization stats saved to: {norm_stats_save_path}")

    print(f"\nConditioning columns ({condition_dim}): {condition_cols}")
    conditional_norm_fn, cond_stats = get_conditional_norm_fn(
        config=norm_cfg["conditions"],
    )
    if condition_cols != cond_stats.cols:
        raise ValueError(
            f"Mismatch between CFM condition_cols {condition_cols} and "
            f"config normalization.conditions.cols {cond_stats.cols}"
        )
    save_conditional_stats(cond_stats, cond_stats_save_path)
    print(f"  Conditional stats saved to: {cond_stats_save_path}")

    # ------------------------------------------------------------------
    # Data loaders: (flux, ivar, mask, condition) 4-tuples
    # ------------------------------------------------------------------
    print("\nCreating data loaders")
    print(f"  invert_mask: {invert_mask}")
    train_loader, val_loader, test_loader = make_loaders(
        dataset_raw,
        nx=nx,
        batch_size=batch_size,
        num_workers=num_workers,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=split_seed,
        image_norm_fn=image_norm_fn,
        return_aux_data=True,
        condition_cols=condition_cols,
        conditional_norm_fn=conditional_norm_fn,
        invert_mask=invert_mask,
        return_noiseless_flux=True,
    )
    print(f"Crop size  : {nx}x{nx} px")
    n_train_batches = (n_train + batch_size - 1) // batch_size
    n_val_batches = (n_val + batch_size - 1) // batch_size
    print(f"Batches    : {n_train_batches} train / {n_val_batches} val")
    if test_loader is not None:
        n_test_batches = (n_test + batch_size - 1) // batch_size
        print(f"           : {n_test_batches} test")

    # ------------------------------------------------------------------
    # Build CFM
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING CFM")
    print("=" * 60)

    cfm = CFM(
        cond_vec_dim=condition_dim,
        in_channels=in_channels,
        input_size=nx,
        base_channels=base_channels,
    ).to(device)
    print(f"CFM parameters: {sum(p.numel() for p in cfm.parameters()):,}")
    print(f"  cond_vec_dim  : {condition_dim}  {condition_cols}")
    print(f"  in_channels   : {in_channels}")
    print(f"  input_size    : {nx}")
    print(f"  base_channels : {base_channels}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    cfm_config = load_cfm_training_config()

    cfm_trainer = CFMTrainer(
        model=cfm,
        train_loader=train_loader,
        config=cfm_config,
        val_loader=val_loader,
        # denorm_fn=image_denorm_fn,
    )
    cfm_trainer.train()

    print("CFM training complete!")

    # ------------------------------------------------------------------
    # Plot loss curves (train + val)
    # ------------------------------------------------------------------
    plot_loss_history(cfm_trainer.loss_history, cfm_trainer.output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    cfm_ckpt = output_dir / "cfm" / "checkpoints"
    print(f"""
Output layout:
  Config file        : {output_dir / "galgenai_config.yaml"}
  Image normalization: {image_norm_type}
  Normalization stats: {norm_stats_save_path}
  Conditional stats  : {cond_stats_save_path}
  CFM checkpoints    : {cfm_ckpt}
  CFM samples        : {output_dir / "cfm" / "samples"}
""")


if __name__ == "__main__":
    main()
