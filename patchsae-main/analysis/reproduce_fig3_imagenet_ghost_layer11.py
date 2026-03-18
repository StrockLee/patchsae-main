import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib import gridspec


DATASET_INFO = {
    "imagenet": {
        "path": "evanarlian/imagenet_1k_resized_256",
        "split": "train",
    }
}


def load_imagenet_classnames(classname_file: Path) -> list[str]:
    lines = [line.strip() for line in classname_file.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    # Format in this repo is usually: "<idx> <class name>"
    return [" ".join(line.split(" ")[1:]) if " " in line else line for line in lines]


def calculate_entropy(top_values: torch.Tensor, top_labels: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Weighted entropy over top activating labels per latent."""
    d_sae = top_labels.shape[0]
    entropy = torch.zeros(d_sae, dtype=torch.float32)
    for latent_idx in range(d_sae):
        labels = top_labels[latent_idx]
        values = top_values[latent_idx]
        unique_labels = labels.unique()
        probs = []
        for label_id in unique_labels:
            probs.append(values[labels == label_id].sum())
        probs = torch.stack(probs)
        probs = probs / (probs.sum() + eps)
        entropy[latent_idx] = -(probs * torch.log(probs + eps)).sum()
    return entropy


def load_feature_tensors(feature_dir: Path) -> dict[str, torch.Tensor]:
    stats = {
        "sparsity": torch.load(feature_dir / "sae_sparsity.pt", map_location="cpu"),
        "mean_acts": torch.load(feature_dir / "sae_mean_acts.pt", map_location="cpu"),
        "top_val": torch.load(feature_dir / "max_activating_image_values.pt", map_location="cpu"),
        "top_idx": torch.load(feature_dir / "max_activating_image_indices.pt", map_location="cpu"),
        "top_label": torch.load(feature_dir / "max_activating_image_label_indices.pt", map_location="cpu"),
    }
    stats["top_idx"] = stats["top_idx"].to(torch.long)
    stats["top_label"] = stats["top_label"].to(torch.long)
    stats["top_entropy"] = calculate_entropy(stats["top_val"], stats["top_label"])
    stats["log_sparsity"] = torch.log10(stats["sparsity"] + 1e-9)
    stats["log_mean_acts"] = torch.log10(stats["mean_acts"] + 1e-9)
    return stats


def top_label_summary(labels_row: torch.Tensor, classnames: list[str]) -> list[dict]:
    unique_labels, counts = labels_row.unique(return_counts=True)
    sorted_idx = torch.argsort(counts, descending=True)
    output = []
    for i in sorted_idx.tolist():
        label_id = int(unique_labels[i].item())
        output.append(
            {
                "label_id": label_id,
                "label_name": classnames[label_id],
                "count": int(counts[i].item()),
            }
        )
    return output


def choose_latents(stats: dict[str, torch.Tensor]) -> dict[str, int]:
    """Heuristic selection for (a)(b)(c)(d) concepts."""
    lab = stats["top_label"]
    ent = stats["top_entropy"]
    x = stats["log_sparsity"]
    y = stats["log_mean_acts"]
    valid = torch.isfinite(x) & torch.isfinite(y) & torch.isfinite(ent)

    def invalidate(score: torch.Tensor) -> torch.Tensor:
        out = score.clone()
        out[~valid] = -1e12
        return out

    # (d) Granny apple
    c_948 = (lab == 948).sum(dim=1).float()
    score_d = c_948 + (c_948 == c_948.max()).float() - 0.02 * torch.abs(x)
    score_d = invalidate(score_d)
    d_idx = int(torch.argmax(score_d).item())

    # (c) comic book style
    c_917 = (lab == 917).sum(dim=1).float()
    # Slightly prefer not-too-trivial entropy while keeping high comic purity.
    score_c = 2.0 * c_917 + 0.15 * ent
    score_c = invalidate(score_c)
    c_idx = int(torch.argmax(score_c).item())

    # (b) text of "MILK": packaging + milk labels
    c_443 = (lab == 443).sum(dim=1).float()
    c_478 = (lab == 478).sum(dim=1).float()
    c_653 = (lab == 653).sum(dim=1).float()
    c_692 = (lab == 692).sum(dim=1).float()
    score_b = 2.8 * c_653 + 1.8 * c_478 + 1.2 * c_692 + 0.8 * c_443 + 0.1 * ent
    # Enforce at least one milk-can label for the MILK concept panel.
    score_b[c_653 < 1] = -1e12
    score_b = invalidate(score_b)
    b_idx = int(torch.argmax(score_b).item())

    # (a) uninterpretable: high entropy in low-frequency/low-activation zone,
    # and weak overlap with a known semantic label.
    set_a = torch.tensor([466, 593, 80, 696, 562, 485], dtype=torch.long)
    overlap_count = torch.isin(lab, set_a).sum(dim=1).float()
    # Unique overlap count via one-hot presence over set_a ids.
    unique_overlap = torch.zeros_like(overlap_count)
    for label_id in set_a.tolist():
        unique_overlap += ((lab == label_id).sum(dim=1) > 0).float()
    region_bonus = -0.6 * torch.abs(x + 2.8) - 0.8 * torch.abs(y + 4.2)
    score_a = 1.3 * ent + 0.8 * unique_overlap + 0.3 * overlap_count + region_bonus
    score_a = invalidate(score_a)
    a_idx = int(torch.argmax(score_a).item())

    # Ensure no duplicates across panels.
    selected = {"a": a_idx, "b": b_idx, "c": c_idx, "d": d_idx}
    used = set()
    for key in ["a", "b", "c", "d"]:
        if selected[key] not in used:
            used.add(selected[key])
            continue
        # Resolve duplicate by taking next-best candidate per key.
        if key == "d":
            candidates = torch.argsort(score_d, descending=True).tolist()
        elif key == "c":
            candidates = torch.argsort(score_c, descending=True).tolist()
        elif key == "b":
            candidates = torch.argsort(score_b, descending=True).tolist()
        else:
            candidates = torch.argsort(score_a, descending=True).tolist()
        for cand in candidates:
            if cand not in used:
                selected[key] = int(cand)
                used.add(int(cand))
                break

    return selected


def plot_scatter_with_marginals(
    stats: dict[str, torch.Tensor],
    selected: dict[str, int],
    out_path: Path,
    show_abcd_labels: bool = True,
    show_label_entropy: bool = True,
    show_label_entropy_hist: bool = True,
    full_axes: bool = False,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    x_all = stats["log_sparsity"]
    y_all = stats["log_mean_acts"]
    c_all = stats["top_entropy"]
    valid = torch.isfinite(x_all) & torch.isfinite(y_all)
    if show_label_entropy:
        valid = valid & torch.isfinite(c_all)
    x = x_all[valid].numpy()
    y = y_all[valid].numpy()
    c = c_all[valid].numpy() if show_label_entropy else None

    if show_label_entropy:
        if show_label_entropy_hist:
            fig = plt.figure(figsize=(13.5, 8), constrained_layout=True)
            gs = gridspec.GridSpec(
                4,
                6,
                figure=fig,
                width_ratios=[4, 4, 4, 1, 0.55, 2.3],
                height_ratios=[2, 4, 4, 4],
            )
            ax_hist_x = fig.add_subplot(gs[0, 0:3])
            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            ax_hist_y = fig.add_subplot(gs[1:4, 3])
            ax_color = fig.add_subplot(gs[:, 4])
            ax_entropy_hist = fig.add_subplot(gs[:, 5])
            sc = ax_scatter.scatter(x, y, c=c, cmap="seismic", s=10, alpha=0.75)
            ax_entropy_hist.hist(c, bins=80, orientation="horizontal", color="#4c5cff", alpha=0.75)
        else:
            fig = plt.figure(figsize=(11.8, 8), constrained_layout=True)
            gs = gridspec.GridSpec(
                4,
                5,
                figure=fig,
                width_ratios=[4, 4, 4, 1, 0.55],
                height_ratios=[2, 4, 4, 4],
            )
            ax_hist_x = fig.add_subplot(gs[0, 0:3])
            ax_scatter = fig.add_subplot(gs[1:4, 0:3])
            ax_hist_y = fig.add_subplot(gs[1:4, 3])
            ax_color = fig.add_subplot(gs[:, 4])
            ax_entropy_hist = None
            sc = ax_scatter.scatter(x, y, c=c, cmap="seismic", s=10, alpha=0.75)
    else:
        fig = plt.figure(figsize=(11.2, 8), constrained_layout=True)
        gs = gridspec.GridSpec(
            4,
            4,
            figure=fig,
            width_ratios=[4, 4, 4, 1],
            height_ratios=[2, 4, 4, 4],
        )
        ax_hist_x = fig.add_subplot(gs[0, 0:3])
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_hist_y = fig.add_subplot(gs[1:4, 3])
        ax_color = None
        ax_entropy_hist = None
        sc = None
        ax_scatter.scatter(x, y, color="#1f4ed8", s=10, alpha=0.6)

    ax_hist_x.hist(x, bins=80, color="#4c5cff", alpha=0.75)
    ax_hist_y.hist(y, bins=80, orientation="horizontal", color="#4c5cff", alpha=0.75)

    # Optional axis clipping (legacy behavior). For full view, keep matplotlib autoscale.
    if not full_axes:
        _, xmax = ax_scatter.get_xlim()
        _, ymax = ax_scatter.get_ylim()
        ax_scatter.set_xlim(left=-4.0, right=xmax)
        ax_scatter.set_ylim(bottom=-5.0, top=ymax)

    # Explicit axis overrides (applied last).
    cur_x_min, cur_x_max = ax_scatter.get_xlim()
    cur_y_min, cur_y_max = ax_scatter.get_ylim()
    ax_scatter.set_xlim(
        left=x_min if x_min is not None else cur_x_min,
        right=x_max if x_max is not None else cur_x_max,
    )
    ax_scatter.set_ylim(
        bottom=y_min if y_min is not None else cur_y_min,
        top=y_max if y_max is not None else cur_y_max,
    )
    ax_hist_x.set_xlim(ax_scatter.get_xlim())
    ax_hist_y.set_ylim(ax_scatter.get_ylim())

    if show_abcd_labels:
        for key, latent_idx in selected.items():
            px = float(stats["log_sparsity"][latent_idx].item())
            py = float(stats["log_mean_acts"][latent_idx].item())
            ax_scatter.scatter([px], [py], s=240, facecolors="none", edgecolors="black", linewidths=2.5, zorder=10)
            ax_scatter.text(px + 0.05, py - 0.08, f"({key})", fontsize=18, weight="bold", color="black")

    ax_scatter.set_xlabel("Log10 Activated Frequency")
    ax_scatter.set_ylabel("Log10 Mean Activation Value")
    ax_hist_x.set_xticks([])
    ax_hist_x.set_yticks([])
    ax_hist_y.set_xticks([])
    ax_hist_y.set_yticks([])
    ax_hist_x.set_frame_on(False)
    ax_hist_y.set_frame_on(False)

    if show_label_entropy and sc is not None:
        cb = fig.colorbar(sc, cax=ax_color)
        cb.set_label("Label Entropy")

        if ax_entropy_hist is not None:
            # Entropy histogram styling.
            ax_entropy_hist.set_title("Label Entropy", fontsize=12, pad=8)
            ax_entropy_hist.set_xlabel("")
            ax_entropy_hist.set_ylabel("")
            ax_entropy_hist.set_xticks([])
            ax_entropy_hist.grid(axis="y", linestyle="--", alpha=0.35)

        # Mark selected (a)(b)(c)(d) entropies on the right histogram.
        if show_abcd_labels and ax_entropy_hist is not None:
            for key, latent_idx in selected.items():
                entropy_val = float(stats["top_entropy"][latent_idx].item())
                if not np.isfinite(entropy_val):
                    continue
                ax_entropy_hist.axhline(entropy_val, color="gray", linestyle=(0, (2, 2)), linewidth=2)
                x_text = ax_entropy_hist.get_xlim()[1] * 0.98
                ax_entropy_hist.text(
                    x_text,
                    entropy_val,
                    f"({key})",
                    va="center",
                    ha="left",
                    fontsize=22,
                    weight="bold",
                    color="black",
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def collect_panel_images(
    dataset,
    stats: dict[str, torch.Tensor],
    latent_idx: int,
    classnames: list[str],
    top_k: int,
):
    indices = stats["top_idx"][latent_idx][:top_k].tolist()
    labels = stats["top_label"][latent_idx][:top_k].tolist()
    panel = []
    for rank, (dataset_idx, label_id) in enumerate(zip(indices, labels), start=1):
        sample = dataset[int(dataset_idx)]
        panel.append(
            {
                "rank": rank,
                "dataset_idx": int(dataset_idx),
                "label_id": int(label_id),
                "label_name": classnames[int(label_id)],
                "image": sample["image"],
            }
        )
    return panel


def draw_four_panels(
    panels: dict[str, list[dict]],
    panel_titles: dict[str, str],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 9))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.06, hspace=0.16)
    order = ["a", "b", "c", "d"]

    for panel_idx, key in enumerate(order):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=outer[panel_idx], wspace=0.03, hspace=0.05
        )
        panel = panels[key]
        for i in range(6):
            ax = fig.add_subplot(inner[i // 3, i % 3])
            ax.imshow(panel[i]["image"])
            ax.set_title(
                f'{panel[i]["label_id"]} {panel[i]["label_name"].split(",")[0][:16]}',
                fontsize=11,
                pad=2,
            )
            ax.axis("off")

        # Put panel caption under each block.
        caption_ax = fig.add_subplot(outer[panel_idx])
        caption_ax.axis("off")
        caption_ax.text(
            0.5,
            -0.07,
            panel_titles[key],
            ha="center",
            va="top",
            fontsize=26 if key == "d" else 25,
            weight="bold" if key in {"a", "b", "c"} else "normal",
            transform=caption_ax.transAxes,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reproduce Figure-3-style plots for imagenet_ghost_layer11")
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="all_direct_product/plot/imagenet_ghost_layer11",
        help="Directory containing sae_sparsity.pt, sae_mean_acts.pt and top-image tensors",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="all_direct_product/plot/imagenet_ghost_layer11/figure3_repro",
    )
    parser.add_argument("--dataset_name", type=str, default="imagenet", choices=["imagenet"])
    parser.add_argument("--seed", type=int, default=1, help="Must match compute_sae_feature_data shuffle seed")
    parser.add_argument("--top_k", type=int, default=6, help="Number of reference images per concept panel")
    parser.add_argument("--skip_reference_images", action="store_true", default=False)
    parser.add_argument("--a_idx", type=int, default=None, help="Manual override for latent (a)")
    parser.add_argument("--b_idx", type=int, default=None, help="Manual override for latent (b)")
    parser.add_argument("--c_idx", type=int, default=None, help="Manual override for latent (c)")
    parser.add_argument("--d_idx", type=int, default=None, help="Manual override for latent (d)")
    parser.add_argument(
        "--hide_abcd_labels",
        action="store_true",
        default=False,
        help="Hide (a)(b)(c)(d) annotations from scatter and entropy histogram.",
    )
    parser.add_argument(
        "--hide_label_entropy",
        action="store_true",
        default=False,
        help="Remove label-entropy colorbar/histogram and use a single scatter color.",
    )
    parser.add_argument(
        "--hide_label_entropy_hist",
        action="store_true",
        default=False,
        help="Hide only the right entropy histogram while keeping entropy color mapping on scatter.",
    )
    parser.add_argument(
        "--full_axes",
        action="store_true",
        default=False,
        help="Show full automatic axis range without clipping x>=-4 and y>=-5.",
    )
    parser.add_argument("--x_min", type=float, default=None, help="Optional x-axis lower bound.")
    parser.add_argument("--x_max", type=float, default=None, help="Optional x-axis upper bound.")
    parser.add_argument("--y_min", type=float, default=None, help="Optional y-axis lower bound.")
    parser.add_argument("--y_max", type=float, default=None, help="Optional y-axis upper bound.")
    parser.add_argument(
        "--classnames_file",
        type=str,
        default="configs/classnames/imagenet_classnames.txt",
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_feature_tensors(feature_dir)
    classnames = load_imagenet_classnames(Path(args.classnames_file))
    selected = choose_latents(stats)

    # Manual override, if provided.
    if args.a_idx is not None:
        selected["a"] = args.a_idx
    if args.b_idx is not None:
        selected["b"] = args.b_idx
    if args.c_idx is not None:
        selected["c"] = args.c_idx
    if args.d_idx is not None:
        selected["d"] = args.d_idx

    # Save selected latent summary.
    summary = {}
    for key, latent_idx in selected.items():
        summary[key] = {
            "latent_idx": int(latent_idx),
            "log10_activated_frequency": float(stats["log_sparsity"][latent_idx].item()),
            "log10_mean_activation": float(stats["log_mean_acts"][latent_idx].item()),
            "label_entropy": float(stats["top_entropy"][latent_idx].item()),
            "top_label_summary": top_label_summary(stats["top_label"][latent_idx], classnames),
        }
    (out_dir / "selected_latents.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Scatter plot with highlighted (a)(b)(c)(d) points.
    plot_scatter_with_marginals(
        stats=stats,
        selected=selected,
        out_path=out_dir / "figure3_scatter.png",
        show_abcd_labels=not args.hide_abcd_labels,
        show_label_entropy=not args.hide_label_entropy,
        show_label_entropy_hist=not args.hide_label_entropy_hist,
        full_axes=args.full_axes,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    if args.skip_reference_images:
        print(json.dumps(summary, indent=2))
        print(f"Saved scatter and latent summary in: {out_dir}")
        return

    # Load shuffled dataset so top_idx maps to the same images used during feature extraction.
    dataset = load_dataset(**DATASET_INFO[args.dataset_name]).shuffle(seed=args.seed)

    panel_titles = {
        "a": "(a) uninterpretable",
        "b": '(b) text of "MILK"',
        "c": "(c) comic book style",
        "d": "(d) Granny apple",
    }
    panels = {
        key: collect_panel_images(
            dataset=dataset,
            stats=stats,
            latent_idx=latent_idx,
            classnames=classnames,
            top_k=args.top_k,
        )
        for key, latent_idx in selected.items()
    }

    draw_four_panels(
        panels=panels,
        panel_titles=panel_titles,
        out_path=out_dir / "figure3_reference_images.png",
    )

    # Also dump panel metadata for reproducibility.
    panel_meta = {}
    for key, panel in panels.items():
        panel_meta[key] = [
            {
                "rank": item["rank"],
                "dataset_idx": item["dataset_idx"],
                "label_id": item["label_id"],
                "label_name": item["label_name"],
            }
            for item in panel
        ]
    (out_dir / "panel_metadata.json").write_text(json.dumps(panel_meta, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs in: {out_dir}")


if __name__ == "__main__":
    main()
