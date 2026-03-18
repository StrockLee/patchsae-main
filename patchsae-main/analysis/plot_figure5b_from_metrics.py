import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY_TOPK = [1, 10, 100, 1000, "all"]


def _infer_row_keys(n_rows: int) -> list[str]:
    """Infer row order for metrics.csv exported by classification_with_top_k_masking.py."""
    if n_rows == 21:
        # no_sae + (on/off for 10 k-values: 1,2,5,10,50,100,500,1000,2000,all)
        topk_full = [1, 2, 5, 10, 50, 100, 500, 1000, 2000, "all"]
    elif n_rows == 11:
        # no_sae + (on/off for 5 k-values: 1,10,100,1000,all)
        topk_full = [1, 10, 100, 1000, "all"]
    else:
        raise ValueError(
            "Unsupported row count for auto-inference. "
            f"Got {n_rows}, expected 11 or 21."
        )

    keys = ["no_sae"]
    for k in topk_full:
        keys.append(f"on_{k}")
        keys.append(f"off_{k}")
    return keys


def load_mean_metrics(csv_path: Path) -> dict[str, float]:
    """Load a wide metrics table and return {metric_key: mean_accuracy}."""
    df = pd.read_csv(csv_path)
    row_keys = _infer_row_keys(len(df))

    if len(row_keys) != len(df):
        raise ValueError(
            f"Row key inference mismatch: inferred {len(row_keys)} keys, got {len(df)} rows"
        )

    mean_by_row = df.mean(axis=1)
    return {k: float(mean_by_row.iloc[i]) for i, k in enumerate(row_keys)}


def extract_curve_points(metric_map: dict[str, float]) -> dict[str, list[float]]:
    """Build off/on points in Figure-5b order."""
    off_vals = [metric_map[f"off_{k}"] for k in DISPLAY_TOPK]
    on_vals = [metric_map[f"on_{k}"] for k in DISPLAY_TOPK]
    return {"off": off_vals, "on": on_vals}


def save_curve_table(
    out_csv: Path,
    class_map: dict[str, float],
    dataset_map: dict[str, float],
    random_map: dict[str, float],
) -> None:
    rows = []
    for strategy, metric_map in [
        ("class-level", class_map),
        ("dataset-level", dataset_map),
        ("random", random_map),
    ]:
        curve = extract_curve_points(metric_map)
        for idx, k in enumerate(DISPLAY_TOPK):
            rows.append(
                {
                    "strategy": strategy,
                    "segment": "off",
                    "topk": str(k),
                    "accuracy": curve["off"][idx],
                }
            )
            rows.append(
                {
                    "strategy": strategy,
                    "segment": "on",
                    "topk": str(k),
                    "accuracy": curve["on"][idx],
                }
            )
        rows.append(
            {
                "strategy": strategy,
                "segment": "no_sae",
                "topk": "baseline",
                "accuracy": float(metric_map["no_sae"]),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)


def plot_figure5b_style(
    class_map: dict[str, float],
    dataset_map: dict[str, float],
    random_map: dict[str, float],
    out_png: Path,
) -> None:
    x_off = [0, 1, 2, 3, 4]
    x_on = [6, 7, 8, 9, 10]
    x_all = x_off + x_on
    tick_labels = ["1", "10", "100", "1000", "all", "1", "10", "100", "1000", "all"]

    class_curve = extract_curve_points(class_map)
    dataset_curve = extract_curve_points(dataset_map)
    random_curve = extract_curve_points(random_map)

    y_class = class_curve["off"] + class_curve["on"]
    y_dataset = dataset_curve["off"] + dataset_curve["on"]
    y_random = random_curve["off"] + random_curve["on"]
    y_no_sae = float(class_map["no_sae"])

    fig, ax = plt.subplots(figsize=(8.6, 6.6), dpi=200)
    bg = "#e7e7e7"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    ax.grid(True, color="#b5b5b5", linewidth=1.3, alpha=0.95)
    for spine in ax.spines.values():
        spine.set_color("#b8b8b8")
        spine.set_linewidth(1.2)

    # random
    ax.plot(
        x_all,
        y_random,
        color="#9a9a9a",
        marker="o",
        markersize=7.8,
        linewidth=3.0,
        label="random",
        alpha=0.75,
        zorder=3,
    )

    # dataset-level
    ax.plot(
        x_all,
        y_dataset,
        color="#1f1f1f",
        marker="o",
        markersize=7.8,
        linewidth=3.0,
        label="dataset-level",
        alpha=0.85,
        zorder=4,
    )

    # class-level
    ax.plot(
        x_all,
        y_class,
        color="#3aac49",
        marker="o",
        markersize=7.8,
        linewidth=3.0,
        label="class-level",
        alpha=0.65,
        zorder=5,
    )

    # no sae baseline
    ax.axhline(
        y_no_sae,
        color="#71c77a",
        linestyle=(0, (1.5, 2.2)),
        linewidth=3.0,
        label="no sae",
        zorder=2,
    )

    # Center separator between "off top-k" and "on top-k"
    y_min = min(min(y_class), min(y_dataset), min(y_random), y_no_sae)
    y_max = max(max(y_class), max(y_dataset), max(y_random), y_no_sae)
    ax.vlines(5, ymin=max(0.0, y_min - 1.0), ymax=min(10.0, y_max), colors="black", linewidth=3.2, zorder=6)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-3.5, max(69.0, y_max + 3.0))
    ax.set_xticks(x_all)
    ax.set_xticklabels(tick_labels, fontsize=14)
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", labelsize=16, length=0)

    ax.set_ylabel("Accuracy (%)", fontsize=18)

    # Group labels
    ax.text(2.0, -0.16, "off top-k", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=18, fontweight="bold")
    ax.text(8.0, -0.16, "on top-k", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=18, fontweight="bold")
    ax.plot([-0.15, 4.15], [-0.135, -0.135], transform=ax.get_xaxis_transform(), color="black", linewidth=1.4, clip_on=False)
    ax.plot([5.85, 10.15], [-0.135, -0.135], transform=ax.get_xaxis_transform(), color="black", linewidth=1.4, clip_on=False)

    # Panel label
    ax.text(-0.15, 1.01, "(b)", transform=ax.transAxes, fontsize=18, fontweight="bold")

    leg = ax.legend(
        loc="lower left",
        framealpha=0.7,
        fancybox=True,
        fontsize=16,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.35,
    )
    leg.get_frame().set_facecolor("#efefef")
    leg.get_frame().set_edgecolor("#c0c0c0")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Figure-5b-style plot from metrics tables. "
            "If dataset/random CSVs are omitted, class CSV is reused."
        )
    )
    parser.add_argument(
        "--class_csv",
        type=str,
        default="all_direct_product/mer/metrics_layer11_imagenet_ghost.csv",
        help="CSV for class-level masking metrics.",
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=None,
        help="CSV for dataset-level masking metrics. Default: reuse class_csv.",
    )
    parser.add_argument(
        "--random_csv",
        type=str,
        default=None,
        help="CSV for random masking metrics. Default: reuse class_csv.",
    )
    parser.add_argument(
        "--out_png",
        type=str,
        default="all_direct_product/mer/figure5b_style.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--out_values_csv",
        type=str,
        default="all_direct_product/mer/figure5b_style_values.csv",
        help="Output aggregated values table.",
    )
    args = parser.parse_args()

    class_csv = Path(args.class_csv)
    dataset_csv = Path(args.dataset_csv) if args.dataset_csv else class_csv
    random_csv = Path(args.random_csv) if args.random_csv else class_csv

    class_map = load_mean_metrics(class_csv)
    dataset_map = load_mean_metrics(dataset_csv)
    random_map = load_mean_metrics(random_csv)

    save_curve_table(Path(args.out_values_csv), class_map, dataset_map, random_map)
    plot_figure5b_style(class_map, dataset_map, random_map, Path(args.out_png))

    print(f"Saved figure: {args.out_png}")
    print(f"Saved values: {args.out_values_csv}")
    if args.dataset_csv is None or args.random_csv is None:
        print(
            "Note: dataset_csv/random_csv not provided, so class_csv was reused. "
            "Provide separate CSVs for non-overlapping curves."
        )


if __name__ == "__main__":
    main()
