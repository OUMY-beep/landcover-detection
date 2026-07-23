"""
predict.py
==========
Génère des prédictions visuelles pour U-Net et SegFormer sur des images du
jeu de test, puis sauvegarde les figures côte à côte dans :

    outputs/predictions/unet/      image001.png  image002.png …
    outputs/predictions/segformer/ image001.png  image002.png …

Chaque figure contient 3 sous-graphiques :
    • Image originale (dénormalisée)
    • Masque Ground Truth
    • Prédiction du modèle

Utilisation :
    python src/evaluation/predict.py               # 5 images, les deux modèles
    python src/evaluation/predict.py --n 10        # 10 images
    python src/evaluation/predict.py --model unet  # U-Net seulement
    python src/evaluation/predict.py --seed 42     # reproductibilité
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # backend non-interactif pour sauvegarder en PNG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Chemins ───────────────────────────────────────────────────────────────────
SRC_ROOT     = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"

# ── Imports internes ──────────────────────────────────────────────────────────
from dataset.loader import datasets
from evaluation.load_models import DEVICE, load_segformer, load_unet
from postprocessing.morphology import apply_morphology
from preprocessing.transforms import MEAN, NUM_CLASSES, STD

# ── Couleurs et noms de classes ───────────────────────────────────────────────
CLASS_NAMES = [
    "Background", "Building", "Road", "Water",
    "Barren",     "Forest",   "Agricultural", "Classe 7",
]

# Palette tab10 (reproductible)
_CMAP_COLORS = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
from matplotlib.colors import ListedColormap
MASK_CMAP = ListedColormap(_CMAP_COLORS)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires image
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Inverse la normalisation ImageNet sur un tenseur (C, H, W).
    Retourne un tableau (H, W, 3) avec des valeurs dans [0, 1].
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def legend_patches() -> list:
    """Retourne la liste de patches de légende pour toutes les classes."""
    return [
        mpatches.Patch(color=_CMAP_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde d'une figure (Original / GT / Prédiction) pour un modèle
# ─────────────────────────────────────────────────────────────────────────────

def save_prediction_figure(
    image:       torch.Tensor,   # (C, H, W) normalisé
    ground_truth: torch.Tensor,  # (H, W) indices de classes
    prediction:  torch.Tensor,   # (H, W) indices de classes
    output_path: Path,
    model_name:  str = "",
    sample_idx:  int = 0,
) -> None:
    """
    Trace et sauvegarde la figure Original / Ground Truth / Prédiction.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1a1a2e")          # fond sombre

    # — Original —
    axes[0].imshow(denormalize(image))
    axes[0].set_title("Image Originale", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # — Ground Truth —
    axes[1].imshow(
        ground_truth.cpu().numpy(),
        cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )
    axes[1].set_title("Ground Truth", color="white", fontsize=13, pad=10)
    axes[1].axis("off")

    # — Prédiction —
    axes[2].imshow(
        prediction.cpu().numpy(),
        cmap=MASK_CMAP, vmin=0, vmax=NUM_CLASSES - 1,
        interpolation="nearest",
    )
    axes[2].set_title(f"Prédiction ({model_name})", color="white", fontsize=13, pad=10)
    axes[2].axis("off")

    # — Légende partagée —
    fig.legend(
        handles=legend_patches(),
        loc="lower center",
        ncol=NUM_CLASSES // 2,
        fontsize=8,
        facecolor="#1a1a2e",
        labelcolor="white",
        framealpha=0.7,
    )

    plt.suptitle(
        f"{model_name}  —  Échantillon #{sample_idx:03d}",
        color="white", fontsize=14, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.10, 1, 1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de prédiction pour un modèle
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_predictions(
    model:      torch.nn.Module,
    dataset,
    indices:    list[int],
    output_dir: Path,
    model_name: str,
    device:     torch.device = DEVICE,
    use_postprocess: bool = False,
) -> None:
    """
    Génère et sauvegarde les figures de prédiction pour les indices donnés.

    Args:
        model:      Modèle de segmentation (en mode eval).
        dataset:    LoveDADataset (jeu de test).
        indices:    Indices des images à traiter.
        output_dir: Répertoire de sortie (ex. outputs/predictions/unet/).
        model_name: Nom affiché dans les titres.
        device:     Périphérique de calcul.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for rank, idx in enumerate(tqdm(indices, desc=f"Prédictions {model_name}")):
        image, mask = dataset[idx]        # (C,H,W), (H,W)

        image_batch = image.unsqueeze(0).to(device)
        logits      = model(image_batch)
        pred        = torch.argmax(logits, dim=1).squeeze(0).cpu()

        if use_postprocess:
            pred_np = apply_morphology(pred.numpy(), road_class=2, building_class=1)
            pred = torch.from_numpy(pred_np)

        out_file = output_dir / f"image{rank + 1:03d}.png"
        save_prediction_figure(
            image        = image,
            ground_truth = mask,
            prediction   = pred,
            output_path  = out_file,
            model_name   = model_name,
            sample_idx   = rank + 1,
        )

    print(f"  → {len(indices)} images sauvegardées dans : {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Générer des prédictions visuelles pour U-Net et/ou SegFormer"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Nombre d'images à traiter (défaut : 5)",
    )
    parser.add_argument(
        "--model", choices=["unet", "segformer", "both"], default="both",
        help="Modèle(s) à utiliser (défaut : both)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Graine aléatoire pour la sélection des images (défaut : 0)",
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Appliquer un lissage morphologique pour corriger les routes cassées",
    )
    args = parser.parse_args()

    print(f"\nPériphérique : {DEVICE}")

    # Sélectionner aléatoirement `n` indices dans le jeu de test
    test_dataset = datasets["test"]
    n_total      = len(test_dataset)
    n_select     = min(args.n, n_total)

    rng = random.Random(args.seed)
    indices = rng.sample(range(n_total), n_select)
    print(f"Images sélectionnées : {indices}")

    # ── U-Net ─────────────────────────────────────────────────────────────────
    if args.model in ("unet", "both"):
        unet = load_unet(device=DEVICE)
        generate_predictions(
            model      = unet,
            dataset    = test_dataset,
            indices    = indices,
            output_dir = PREDICTIONS_DIR / "unet",
            model_name = "U-Net",
            device     = DEVICE,
            use_postprocess = args.postprocess,
        )
        del unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── SegFormer ─────────────────────────────────────────────────────────────
    if args.model in ("segformer", "both"):
        segformer = load_segformer(device=DEVICE)
        generate_predictions(
            model      = segformer,
            dataset    = test_dataset,
            indices    = indices,
            output_dir = PREDICTIONS_DIR / "segformer",
            model_name = "SegFormer",
            device     = DEVICE,
            use_postprocess = args.postprocess,
        )
        del segformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nToutes les prédictions sont dans : {PREDICTIONS_DIR}\n")


if __name__ == "__main__":
    main()