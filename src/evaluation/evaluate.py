"""
evaluate.py
===========
Évalue U-Net et SegFormer sur le jeu de test LoveDA.

Métriques calculées pour chaque modèle :
    - Test Loss (CrossEntropyLoss)
    - Pixel Accuracy
    - Mean IoU
    - Mean Dice / F1 Score
    - Precision par classe
    - Recall par classe
    - F1 par classe (= Dice par classe)
    - IoU par classe
    - FPS  (= 1000 / ms_par_image)
    - Taille mémoire du modèle (MB)
    - Matrice de confusion (NUM_CLASSES × NUM_CLASSES)

Sortie :
    - Rapport console détaillé
    - outputs/reports/metrics_unet.json
    - outputs/reports/metrics_segformer.json
    - outputs/reports/confusion_unet.png       (heatmap normalisée)
    - outputs/reports/confusion_unet.npy       (matrice brute NumPy)
    - outputs/reports/confusion_unet.csv       (matrice brute CSV)
    - outputs/reports/confusion_segformer.png / .npy / .csv
    - outputs/reports/model_comparison.csv  (mis à jour par ce script)

Utilisation :
    python src/evaluation/evaluate.py
    python src/evaluation/evaluate.py --model unet
    python src/evaluation/evaluate.py --model segformer
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
SRC_ROOT     = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports internes ──────────────────────────────────────────────────────────
from dataset.loader import loaders
from evaluation.load_models import DEVICE, load_segformer, load_unet
from preprocessing.transforms import NUM_CLASSES
from training.metrics import (
    dice_per_class,
    intersection_and_union,
    iou_per_class,
    mean_dice,
    mean_iou,
    pixel_accuracy,
)

# ── Constantes ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Background", "Building", "Road", "Water",
    "Barren", "Forest", "Agricultural", "Classe 7",
]

CRITERION    = nn.CrossEntropyLoss(ignore_index=255)
IGNORE_INDEX = 255


# ─────────────────────────────────────────────────────────────────────────────
# Taille mémoire du modèle
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    """
    Calcule la taille des paramètres + buffers du modèle en mégaoctets.
    Équivalent à la taille du fichier .pth (poids uniquement).
    """
    total_bytes = sum(
        p.nelement() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    )
    return total_bytes / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Matrice de confusion
# ─────────────────────────────────────────────────────────────────────────────

def update_confusion_matrix(
    conf_matrix: torch.Tensor,
    logits:      torch.Tensor,
    masks:       torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> None:
    """
    Met à jour la matrice de confusion en place (num_classes × num_classes).
    Axe 0 = classes réelles (ground truth), Axe 1 = classes prédites.
    Les pixels avec mask == ignore_index sont ignorés.
    """
    preds = torch.argmax(logits, dim=1).cpu().view(-1)
    gts   = masks.cpu().view(-1)

    valid = gts != ignore_index
    preds = preds[valid]
    gts   = gts[valid]

    # Encodage linéaire : (gt * C + pred) → histogramme
    combined = (gts * num_classes + preds).long()
    counts   = torch.bincount(combined, minlength=num_classes ** 2)
    conf_matrix += counts.view(num_classes, num_classes).float()


def precision_per_class(
    conf_matrix: np.ndarray,
) -> np.ndarray:
    """
    Précision par classe depuis la matrice de confusion brute.
    Precision_c = TP_c / (TP_c + FP_c)  = cm[c,c] / cm[:,c].sum()
    Renvoie NaN pour les classes jamais prédites.
    """
    col_sums = conf_matrix.sum(axis=0)          # prédit comme classe c
    diag     = np.diag(conf_matrix)
    prec     = np.where(col_sums > 0, diag / col_sums, np.nan)
    return prec


def recall_per_class(
    conf_matrix: np.ndarray,
) -> np.ndarray:
    """
    Rappel par classe depuis la matrice de confusion brute.
    Recall_c = TP_c / (TP_c + FN_c)  = cm[c,c] / cm[c,:].sum()
    Renvoie NaN pour les classes absentes du GT.
    """
    row_sums = conf_matrix.sum(axis=1)          # GT de classe c
    diag     = np.diag(conf_matrix)
    rec      = np.where(row_sums > 0, diag / row_sums, np.nan)
    return rec


def save_confusion_matrix(
    conf_matrix: torch.Tensor,
    output_path: Path,
    model_name:  str = "",
) -> None:
    """
    Sauvegarde la matrice de confusion sous trois formats :
      - PNG  : heatmap normalisée par ligne (recall)
      - .npy : matrice brute (int counts) pour analyses ultérieures
      - .csv : matrice brute avec en-têtes de classes
    """
    cm_np = conf_matrix.numpy().astype(np.float64)

    # ── .npy brut ─────────────────────────────────────────────────────────────
    npy_path = output_path.with_suffix(".npy")
    np.save(npy_path, cm_np)
    print(f"  → Confusion brute (.npy) : {npy_path}")

    # ── .csv brut ─────────────────────────────────────────────────────────────
    import csv as _csv
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow([""] + CLASS_NAMES)          # en-tête colonnes
        for i, row in enumerate(cm_np):
            writer.writerow(
                [CLASS_NAMES[i]] + [str(int(v)) for v in row]
            )
    print(f"  → Confusion brute (.csv) : {csv_path}")

    # ── PNG normalisé ─────────────────────────────────────────────────────────
    row_sums = cm_np.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_np / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val   = cm_norm[i, j]
            color = "white" if val < 0.5 else "#0d1117"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(CLASS_NAMES, color="white", fontsize=9)
    ax.set_xlabel("Classe prédite",     color="white", fontsize=11, labelpad=10)
    ax.set_ylabel("Classe réelle (GT)", color="white", fontsize=11, labelpad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    title = f"Matrice de Confusion — {model_name}" if model_name else "Matrice de Confusion"
    ax.set_title(title, color="white", fontsize=13, pad=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → Confusion (PNG)        : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Boucle d'évaluation principale
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model:      nn.Module,
    loader,
    device:     torch.device = DEVICE,
    model_name: str = "model",
) -> dict:
    """
    Calcule toutes les métriques de segmentation sur un DataLoader.

    Utilise torch.inference_mode() : plus rapide et plus léger que no_grad()
    (désactive aussi l'autograd pour les vues de tenseurs).

    Returns:
        dict avec les clés :
            test_loss, pixel_accuracy, mean_iou, mean_dice (= mean_f1),
            iou_per_class, dice_per_class (= f1_per_class),
            precision_per_class, recall_per_class,
            num_parameters, model_size_mb,
            inference_time_ms, fps,
            confusion_matrix (list[list])
    """
    model.eval()

    running_loss   = 0.0
    running_acc    = 0.0
    total_samples  = 0
    total_inter    = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_union    = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total_time_ms  = 0.0
    conf_matrix    = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float32)

    with torch.inference_mode():   # ← plus efficace que @no_grad
        for images, masks in tqdm(loader, desc=f"Évaluation {model_name}"):
            images = images.to(device)
            masks  = masks.to(device)

            # ── Chronomètre ───────────────────────────────────────────────────
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0     = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1     = time.perf_counter()
            total_time_ms += (t1 - t0) * 1000.0

            # ── Métriques ─────────────────────────────────────────────────────
            running_loss  += CRITERION(logits, masks).item() * images.size(0)
            running_acc   += pixel_accuracy(logits, masks) * images.size(0)
            total_samples += images.size(0)

            batch_inter, batch_union = intersection_and_union(logits, masks)
            total_inter += batch_inter
            total_union += batch_union

            # ── Matrice de confusion ──────────────────────────────────────────
            update_confusion_matrix(conf_matrix, logits, masks)

    iou  = iou_per_class(total_inter, total_union)
    dice = dice_per_class(total_inter, total_union)

    avg_ms  = total_time_ms / max(total_samples, 1)
    fps_val = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    n_params = sum(p.numel() for p in model.parameters())
    size_mb  = model_size_mb(model)

    # ── Precision / Recall depuis la matrice de confusion ─────────────────────
    cm_np  = conf_matrix.numpy().astype(np.float64)
    prec   = precision_per_class(cm_np)   # shape (NUM_CLASSES,)
    rec    = recall_per_class(cm_np)      # shape (NUM_CLASSES,)

    def _fmt(arr: np.ndarray) -> list:
        return [float(v) if not np.isnan(v) else None for v in arr]

    results = {
        "model_name"        : model_name,
        "test_loss"         : running_loss / max(total_samples, 1),
        "pixel_accuracy"    : running_acc  / max(total_samples, 1),
        "mean_iou"          : mean_iou(iou),
        "mean_dice"         : mean_dice(dice),   # = mean F1
        "iou_per_class"     : [v.item() if not torch.isnan(v) else None for v in iou],
        "dice_per_class"    : [v.item() if not torch.isnan(v) else None for v in dice],
        "precision_per_class": _fmt(prec),
        "recall_per_class"  : _fmt(rec),
        "num_parameters"    : n_params,
        "model_size_mb"     : round(size_mb, 2),
        "inference_time_ms" : round(avg_ms,  2),
        "fps"               : round(fps_val, 1),
        "confusion_matrix"  : conf_matrix.tolist(),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Affichage console
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """Affiche un rapport complet dans la console."""
    sep  = "=" * 70
    sep2 = "-" * 70

    print(f"\n{sep}")
    print(f"  ÉVALUATION SUR LE JEU DE TEST — {results['model_name'].upper()}")
    print(sep)
    print(f"  Test Loss              : {results['test_loss']:.4f}")
    print(f"  Pixel Accuracy         : {results['pixel_accuracy'] * 100:.2f}%")
    print(f"  Mean IoU               : {results['mean_iou']  * 100:.2f}%")
    print(f"  Mean Dice / F1 Score   : {results['mean_dice'] * 100:.2f}%")
    print(f"  Paramètres             : {results['num_parameters']:,}")
    print(f"  Taille mémoire         : {results['model_size_mb']:.1f} MB")
    print(f"  Temps inférence        : {results['inference_time_ms']:.1f} ms/image")
    print(f"  FPS                    : {results['fps']:.1f} img/s")
    print()
    print("  IoU / F1 / Precision / Recall par classe")
    print(sep2)
    header_cls = f"  {'Classe':15s}  {'IoU':>7s}  {'F1/Dice':>8s}  {'Precision':>10s}  {'Recall':>7s}"
    print(header_cls)
    print("  " + "-" * 58)

    for i, name in enumerate(CLASS_NAMES):
        iou_val  = results["iou_per_class"][i]
        dice_val = results["dice_per_class"][i]
        prec_val = results["precision_per_class"][i]
        rec_val  = results["recall_per_class"][i]

        def _pct(v): return f"{v*100:.1f}%" if v is not None else "  N/A "
        print(
            f"  {name:15s}  {_pct(iou_val):>7s}  {_pct(dice_val):>8s}"
            f"  {_pct(prec_val):>10s}  {_pct(rec_val):>7s}"
        )

    print(sep)

    # ── Top confusions ────────────────────────────────────────────────────────
    cm = np.array(results["confusion_matrix"])
    print("\n  Top 5 confusions (GT → Prédit, hors diagonale)")
    print(sep2)
    np.fill_diagonal(cm, 0)
    flat = cm.flatten()
    top5 = np.argsort(flat)[::-1][:5]
    for idx in top5:
        gt_cls   = idx // NUM_CLASSES
        pred_cls = idx  % NUM_CLASSES
        count    = flat[idx]
        if count > 0:
            print(f"  {CLASS_NAMES[gt_cls]:15s} → {CLASS_NAMES[pred_cls]:15s}  ({int(count):,} pixels)")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics_json(results: dict, output_path: Path) -> None:
    """Sérialise le dict de métriques en JSON (sans la matrice de confusion)."""
    export = {k: v for k, v in results.items() if k != "confusion_matrix"}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  → Métriques sauvegardées : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tableau de comparaison CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_csv(results_dict: dict[str, dict]) -> None:
    """
    Génère outputs/reports/model_comparison.csv incluant FPS et taille mémoire.

    Args:
        results_dict: {"unet": {...}, "segformer": {...}}
    """
    import csv

    csv_path = REPORTS_DIR / "model_comparison.csv"

    u = results_dict["unet"]
    s = results_dict["segformer"]

    header = ("Metric", "U-Net", "SegFormer")
    global_rows = [
        header,
        ("Test Loss",               f"{u['test_loss']:.4f}",               f"{s['test_loss']:.4f}"),
        ("Pixel Accuracy",          f"{u['pixel_accuracy']*100:.2f}%",      f"{s['pixel_accuracy']*100:.2f}%"),
        ("Mean IoU",                f"{u['mean_iou']*100:.2f}%",            f"{s['mean_iou']*100:.2f}%"),
        ("Mean Dice / F1 Score",    f"{u['mean_dice']*100:.2f}%",           f"{s['mean_dice']*100:.2f}%"),
        ("Parameters",              f"{u['num_parameters']:,}",             f"{s['num_parameters']:,}"),
        ("Model Size (MB)",         f"{u['model_size_mb']:.1f} MB",         f"{s['model_size_mb']:.1f} MB"),
        ("Inference Time (ms/img)", f"{u['inference_time_ms']:.1f} ms",     f"{s['inference_time_ms']:.1f} ms"),
        ("FPS",                     f"{u['fps']:.1f}",                      f"{s['fps']:.1f}"),
    ]

    def _pct(v): return f"{v*100:.2f}%" if v is not None else "N/A"

    iou_rows  = [("", "", ""), ("IoU par classe",       "U-Net", "SegFormer")]
    f1_rows   = [("", "", ""), ("F1 (Dice) par classe", "U-Net", "SegFormer")]
    prec_rows = [("", "", ""), ("Precision par classe",  "U-Net", "SegFormer")]
    rec_rows  = [("", "", ""), ("Recall par classe",     "U-Net", "SegFormer")]

    for i, name in enumerate(CLASS_NAMES):
        iou_rows.append( (f"  IoU  {name}", _pct(u["iou_per_class"][i]),      _pct(s["iou_per_class"][i])))
        f1_rows.append(  (f"  F1   {name}", _pct(u["dice_per_class"][i]),     _pct(s["dice_per_class"][i])))
        prec_rows.append((f"  Prec {name}", _pct(u["precision_per_class"][i]),_pct(s["precision_per_class"][i])))
        rec_rows.append( (f"  Rec  {name}", _pct(u["recall_per_class"][i]),   _pct(s["recall_per_class"][i])))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(global_rows)
        writer.writerows(iou_rows)
        writer.writerows(f1_rows)
        writer.writerows(prec_rows)
        writer.writerows(rec_rows)

    print(f"\n  → Tableau de comparaison : {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation U-Net / SegFormer sur le jeu de test LoveDA"
    )
    parser.add_argument(
        "--model",
        choices=["unet", "segformer", "both"],
        default="both",
        help="Modèle(s) à évaluer (défaut : both)",
    )
    args = parser.parse_args()

    print(f"\nPériphérique : {DEVICE}")
    test_loader  = loaders["test"]
    results_dict: dict[str, dict] = {}

    # ── U-Net ────────────────────────────────────────────────────────────────
    if args.model in ("unet", "both"):
        unet     = load_unet(device=DEVICE)
        res_unet = evaluate_model(unet, test_loader, DEVICE, model_name="U-Net")
        print_report(res_unet)
        save_metrics_json(res_unet, REPORTS_DIR / "metrics_unet.json")
        save_confusion_matrix(
            torch.tensor(res_unet["confusion_matrix"]),
            REPORTS_DIR / "confusion_unet.png",
            model_name="U-Net",
        )
        results_dict["unet"] = res_unet
        del unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── SegFormer ────────────────────────────────────────────────────────────
    if args.model in ("segformer", "both"):
        segformer = load_segformer(device=DEVICE)
        res_seg   = evaluate_model(segformer, test_loader, DEVICE, model_name="SegFormer")
        print_report(res_seg)
        save_metrics_json(res_seg, REPORTS_DIR / "metrics_segformer.json")
        save_confusion_matrix(
            torch.tensor(res_seg["confusion_matrix"]),
            REPORTS_DIR / "confusion_segformer.png",
            model_name="SegFormer",
        )
        results_dict["segformer"] = res_seg
        del segformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── CSV de comparaison ────────────────────────────────────────────────────
    if "unet" in results_dict and "segformer" in results_dict:
        save_comparison_csv(results_dict)

    print("\nÉvaluation terminée.\n")


if __name__ == "__main__":
    main()