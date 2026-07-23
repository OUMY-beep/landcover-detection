"""
compare_models.py
=================
Analyse qualitative côte à côte :

    Original → Ground Truth → Prédiction U-Net → Prédiction SegFormer

pour plusieurs images du jeu de test.

Chaque figure (5 lignes × 4 colonnes) affiche :
    Ligne principale (grande) :
        • Image originale
        • Ground Truth
        • Prédiction U-Net    (avec mIoU image)
        • Prédiction SegFormer (avec mIoU image)
    Ligne inférieure (petite) :
        • Nom de fichier de l'image
        • Distribution GT (% pixels par classe)
        • Distribution U-Net
        • Distribution SegFormer

Titre global : δ mIoU entre les deux modèles + modèle gagnant.

Sorties :
    outputs/predictions/comparison/image001.png
    outputs/predictions/comparison/image002.png
    …

Utilisation :
    python src/evaluation/compare_models.py               # 5 images
    python src/evaluation/compare_models.py --n 8         # 8 images
    python src/evaluation/compare_models.py --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
SRC_ROOT     = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

COMPARISON_DIR = PROJECT_ROOT / "outputs" / "predictions" / "comparison"

# ── Imports internes ──────────────────────────────────────────────────────────
from dataset.loader import datasets
from evaluation.load_models import DEVICE, load_both
from preprocessing.transforms import MEAN, NUM_CLASSES, STD

# ── Palette de couleurs ───────────────────────────────────────────────────────
CLASS_NAMES  = [
    "Background", "Building", "Road", "Water",
    "Barren",     "Forest",   "Agricultural", "Classe 7",
]
CLASS_SHORT  = ["BG", "Bldg", "Road", "Water", "Barren", "Forest", "Agri", "C7"]

_CMAP_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
MASK_CMAP    = ListedColormap(_CMAP_COLORS)

# Fond sombre
BG_COLOR     = "#0d1117"
TITLE_COLOR  = "#e6edf3"
SUBTITLE_CLR = "#8b949e"
UNET_COLOR   = "#58a6ff"   # bleu
SEG_COLOR    = "#3fb950"   # vert
GT_COLOR     = "#f0883e"   # orange


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Inverse la normalisation ImageNet : (C,H,W) → (H,W,3) dans [0,1]."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def legend_patches() -> list:
    return [
        mpatches.Patch(color=_CMAP_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]


def class_distribution(mask: torch.Tensor, ignore: int = 255) -> np.ndarray:
    """
    Retourne le pourcentage de pixels pour chaque classe (ignore `ignore`).
    Shape : (NUM_CLASSES,)
    """
    valid = mask[mask != ignore]
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for cls in range(NUM_CLASSES):
        counts[cls] = (valid == cls).sum().item()
    total = counts.sum()
    return (counts / total * 100.0) if total > 0 else counts


def compute_image_miou(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    ignore: int = 255,
) -> float:
    """mIoU sur une seule image (classes présentes), en pourcentage."""
    valid  = gt != ignore
    pred_v = pred[valid]
    gt_v   = gt[valid]
    ious   = []
    for cls in range(NUM_CLASSES):
        inter = ((pred_v == cls) & (gt_v == cls)).sum().item()
        union = ((pred_v == cls) | (gt_v == cls)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return (sum(ious) / len(ious) * 100.0) if ious else 0.0


def compute_iou_per_class(
    pred: torch.Tensor,
    gt:   torch.Tensor,
    ignore: int = 255,
) -> np.ndarray:
    """
    IoU par classe pour une image. NaN si classe absente de GT et de pred.
    Shape : (NUM_CLASSES,)
    """
    valid  = gt != ignore
    pred_v = pred[valid]
    gt_v   = gt[valid]
    ious   = np.full(NUM_CLASSES, np.nan)
    for cls in range(NUM_CLASSES):
        inter = ((pred_v == cls) & (gt_v == cls)).sum().item()
        union = ((pred_v == cls) | (gt_v == cls)).sum().item()
        if union > 0:
            ious[cls] = inter / union
    return ious


# ─────────────────────────────────────────────────────────────────────────────
# Mini graphique de distribution de classes
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    ax,
    dist:   np.ndarray,
    title:  str,
    color:  str,
) -> None:
    """
    Trace un histogramme horizontal de la distribution de classes dans `ax`.
    """
    ax.set_facecolor(BG_COLOR)
    y_pos = np.arange(NUM_CLASSES)
    ax.barh(y_pos, dist, color=[_CMAP_COLORS[i] for i in range(NUM_CLASSES)],
            edgecolor="#30363d", linewidth=0.5, height=0.7)

    # Annotations des valeurs > 1%
    for i, val in enumerate(dist):
        if val >= 1.0:
            ax.text(val + 0.5, i, f"{val:.1f}%",
                    va="center", ha="left", fontsize=6, color=TITLE_COLOR)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_SHORT, fontsize=6, color=TITLE_COLOR)
    ax.set_xlim(0, max(dist.max() * 1.2, 10))
    ax.set_xlabel("%", fontsize=7, color=SUBTITLE_CLR, labelpad=2)
    ax.tick_params(axis="x", labelsize=6, colors=SUBTITLE_CLR)
    ax.tick_params(axis="y", colors=TITLE_COLOR)
    ax.set_title(title, color=color, fontsize=8, pad=4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.xaxis.grid(True, color="#30363d", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure de comparaison enrichie (GridSpec)
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_figure(
    image:        torch.Tensor,   # (C, H, W)
    ground_truth: torch.Tensor,   # (H, W)
    pred_unet:    torch.Tensor,   # (H, W)
    pred_seg:     torch.Tensor,   # (H, W)
    output_path:  Path,
    sample_idx:   int   = 0,
    miou_unet:    float = 0.0,
    miou_seg:     float = 0.0,
    image_name:   str   = "",
    iou_diff:     np.ndarray | None = None,   # (NUM_CLASSES,) SegFormer-UNet
) -> None:
    """
    Sauvegarde une figure enrichie :

    Rangée haute (maps seg) :
        Original | Ground Truth | U-Net (mIoU) | SegFormer (mIoU)

    Rangée basse (distributions + infos) :
        Nom de fichier | Distrib. GT | Distrib. U-Net | Distrib. SegFormer
    """
    # GridSpec : 2 rangées — grande (maps) + petite (distributions)
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor(BG_COLOR)

    gs = GridSpec(
        2, 4,
        figure=fig,
        height_ratios=[3, 1.4],
        hspace=0.35,
        wspace=0.12,
    )

    ax_img  = fig.add_subplot(gs[0, 0])
    ax_gt   = fig.add_subplot(gs[0, 1])
    ax_unet = fig.add_subplot(gs[0, 2])
    ax_seg  = fig.add_subplot(gs[0, 3])

    ax_info  = fig.add_subplot(gs[1, 0])
    ax_dgt   = fig.add_subplot(gs[1, 1])
    ax_dunet = fig.add_subplot(gs[1, 2])
    ax_dseg  = fig.add_subplot(gs[1, 3])

    # ── Rangée haute : cartes de segmentation ────────────────────────────────

    ax_img.imshow(denormalize(image))
    ax_img.set_title("Image Originale", color=TITLE_COLOR, fontsize=12, pad=8)
    ax_img.axis("off")

    ax_gt.imshow(ground_truth.cpu().numpy(),
                 cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax_gt.set_title("Ground Truth", color=GT_COLOR, fontsize=12, pad=8)
    ax_gt.axis("off")

    ax_unet.imshow(pred_unet.cpu().numpy(),
                   cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax_unet.set_title(f"U-Net\nmIoU : {miou_unet:.1f}%",
                      color=UNET_COLOR, fontsize=12, pad=8, linespacing=1.5)
    ax_unet.axis("off")

    ax_seg.imshow(pred_seg.cpu().numpy(),
                  cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
    ax_seg.set_title(f"SegFormer\nmIoU : {miou_seg:.1f}%",
                     color=SEG_COLOR, fontsize=12, pad=8, linespacing=1.5)
    ax_seg.axis("off")

    # ── Rangée basse : infos + distributions ─────────────────────────────────

    # Panneau infos (nom de fichier + IoU diff par classe)
    ax_info.set_facecolor(BG_COLOR)
    ax_info.axis("off")

    winner      = "U-Net" if miou_unet >= miou_seg else "SegFormer"
    winner_clr  = UNET_COLOR if miou_unet >= miou_seg else SEG_COLOR
    delta       = abs(miou_unet - miou_seg)

    info_lines = [
        f"Fichier : {image_name or 'N/A'}",
        f"Échantillon #{sample_idx:03d}",
        "",
        f"Δ mIoU : {delta:.1f}%",
        f"→ {winner} gagne",
    ]
    # IoU diff par classe (SegFormer − U-Net)
    if iou_diff is not None:
        info_lines += ["", "ΔIoU/classe (Seg−UNet):"]
        for cls_i, diff in enumerate(iou_diff):
            if not np.isnan(diff):
                sign = "+" if diff >= 0 else ""
                info_lines.append(f"  {CLASS_SHORT[cls_i]:6s}: {sign}{diff*100:.1f}%")

    y_start = 0.97
    for line in info_lines:
        color = winner_clr if "gagne" in line else (SUBTITLE_CLR if line.startswith("Δ") else TITLE_COLOR)
        ax_info.text(0.05, y_start, line,
                     transform=ax_info.transAxes,
                     color=color, fontsize=7.5, va="top",
                     fontweight="bold" if "gagne" in line else "normal")
        y_start -= 0.115

    # Distributions
    dist_gt   = class_distribution(ground_truth)
    dist_unet = class_distribution(pred_unet)
    dist_seg  = class_distribution(pred_seg)

    plot_class_distribution(ax_dgt,   dist_gt,   "Distribution GT",        GT_COLOR)
    plot_class_distribution(ax_dunet, dist_unet,  "Distribution U-Net",     UNET_COLOR)
    plot_class_distribution(ax_dseg,  dist_seg,   "Distribution SegFormer", SEG_COLOR)

    # ── Titre global ──────────────────────────────────────────────────────────
    title_main = (
        f"Comparaison U-Net vs SegFormer  —  Image #{sample_idx:03d}\n"
        f"U-Net {miou_unet:.1f}%  |  SegFormer {miou_seg:.1f}%"
        f"  →  Δ {delta:.1f}% en faveur de {winner}"
    )
    fig.suptitle(title_main, color=TITLE_COLOR, fontsize=12, y=0.99, linespacing=1.6)

    # ── Légende partagée ──────────────────────────────────────────────────────
    fig.legend(
        handles=legend_patches(),
        loc="lower center",
        ncol=NUM_CLASSES,
        fontsize=7.5,
        facecolor=BG_COLOR,
        labelcolor=TITLE_COLOR,
        framealpha=0.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    models:  dict,
    dataset,
    indices: list[int],
    device:  torch.device = DEVICE,
) -> None:
    """
    Génère une figure de comparaison pour chaque indice sélectionné.

    Args:
        models:  {"unet": UNet, "segformer": SegFormer}
        dataset: LoveDADataset (jeu de test)
        indices: Indices des images à traiter
        device:  Périphérique de calcul
    """
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    unet      = models["unet"].eval()
    segformer = models["segformer"].eval()

    scores_summary: list[dict] = []

    with torch.inference_mode():
        for rank, idx in enumerate(tqdm(indices, desc="Comparaison qualitative")):
            image, mask = dataset[idx]

            # Nom de fichier de l'image source
            try:
                img_path  = dataset.images_dir / dataset.image_files[idx]
                img_name  = img_path.name
            except (AttributeError, IndexError):
                img_name  = f"idx_{idx}"

            img_batch = image.unsqueeze(0).to(device)

            # ── Prédictions ───────────────────────────────────────────────────
            pred_u = torch.argmax(unet(img_batch),      dim=1).squeeze(0).cpu()
            pred_s = torch.argmax(segformer(img_batch), dim=1).squeeze(0).cpu()

            # ── mIoU par image ────────────────────────────────────────────────
            miou_u = compute_image_miou(pred_u, mask)
            miou_s = compute_image_miou(pred_s, mask)

            # ── IoU par classe → diff SegFormer − U-Net ───────────────────────
            iou_u   = compute_iou_per_class(pred_u, mask)
            iou_s   = compute_iou_per_class(pred_s, mask)
            iou_diff = np.where(
                ~np.isnan(iou_u) & ~np.isnan(iou_s),
                iou_s - iou_u,
                np.nan,
            )

            scores_summary.append({
                "image"         : rank + 1,
                "name"          : img_name,
                "miou_unet"     : round(miou_u, 2),
                "miou_segformer": round(miou_s, 2),
            })

            out_file = COMPARISON_DIR / f"image{rank + 1:03d}.png"
            save_comparison_figure(
                image        = image,
                ground_truth = mask,
                pred_unet    = pred_u,
                pred_seg     = pred_s,
                output_path  = out_file,
                sample_idx   = rank + 1,
                miou_unet    = miou_u,
                miou_seg     = miou_s,
                image_name   = img_name,
                iou_diff     = iou_diff,
            )

    # ── Résumé console ────────────────────────────────────────────────────────
    sep = "-" * 62
    print(f"\n  {'Image':<8} {'Fichier':<22} {'U-Net':>9}  {'SegFormer':>10}  {'Meilleur':>10}")
    print("  " + sep)
    for sc in scores_summary:
        winner = "U-Net" if sc["miou_unet"] >= sc["miou_segformer"] else "SegFormer"
        print(
            f"  #{sc['image']:<6}  {sc['name']:<22}"
            f"  {sc['miou_unet']:>7.1f}%  {sc['miou_segformer']:>8.1f}%  {winner:>10}"
        )

    avg_u = sum(s["miou_unet"]      for s in scores_summary) / max(len(scores_summary), 1)
    avg_s = sum(s["miou_segformer"] for s in scores_summary) / max(len(scores_summary), 1)
    print("  " + sep)
    print(f"  {'Moyenne':<8}  {'':22}  {avg_u:>7.1f}%  {avg_s:>8.1f}%")

    print(f"\n  → {len(indices)} figures sauvegardées dans : {COMPARISON_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse qualitative côte à côte U-Net vs SegFormer"
    )
    parser.add_argument("--n",    type=int, default=5,  help="Nombre d'images (défaut : 5)")
    parser.add_argument("--seed", type=int, default=0,  help="Graine aléatoire (défaut : 0)")
    args = parser.parse_args()

    print(f"\nPériphérique : {DEVICE}")

    # ── Sélection aléatoire d'images du jeu de test ───────────────────────────
    test_dataset = datasets["test"]
    n_total      = len(test_dataset)
    n_select     = min(args.n, n_total)

    rng     = random.Random(args.seed)
    indices = rng.sample(range(n_total), n_select)
    print(f"Images sélectionnées : {indices}")

    # ── Chargement des deux modèles ───────────────────────────────────────────
    models = load_both(device=DEVICE)

    # ── Génération des figures ────────────────────────────────────────────────
    run_comparison(models, test_dataset, indices, DEVICE)

    print("\nAnalyse qualitative terminée.\n")


if __name__ == "__main__":
    main()
