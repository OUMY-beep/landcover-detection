"""
load_models.py
==============
Chargement des meilleurs modèles entraînés depuis outputs/models/.

Utilisation rapide :
    from evaluation.load_models import load_unet, load_segformer, load_both

    unet      = load_unet()
    segformer = load_segformer()
    models    = load_both()   # {"unet": ..., "segformer": ...}
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# ── Chemins ──────────────────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

PROJECT_ROOT  = SRC_ROOT.parent
MODELS_DIR    = PROJECT_ROOT / "outputs" / "models"

UNET_CKPT      = MODELS_DIR / "unet_best.pth"
SEGFORMER_CKPT = MODELS_DIR / "segformer_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Imports internes ──────────────────────────────────────────────────────────
from models.unet.unet               import UNet
from models.segformer.segformer     import SegFormer
from preprocessing.transforms       import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Migration des poids SegFormer (transformers v4.x -> v5.x)
# ─────────────────────────────────────────────────────────────────────────────

def migrate_segformer_keys(state_dict: dict) -> dict:
    """
    Migre les clés de l'ancien format `transformers` (utilisant `encoder`)
    vers le nouveau format (utilisant `stages`), suite à une mise à jour de la librairie.
    """
    # Vérification rapide : si le nouveau format est déjà là, on ne touche à rien
    if any(k.startswith("model.segformer.stages.") for k in state_dict.keys()):
        return state_dict

    new_sd = {}
    for k, v in state_dict.items():
        if "encoder.patch_embeddings.0." in k: k = k.replace("encoder.patch_embeddings.0.", "stages.0.patch_embeddings.")
        elif "encoder.patch_embeddings.1." in k: k = k.replace("encoder.patch_embeddings.1.", "stages.1.patch_embeddings.")
        elif "encoder.patch_embeddings.2." in k: k = k.replace("encoder.patch_embeddings.2.", "stages.2.patch_embeddings.")
        elif "encoder.patch_embeddings.3." in k: k = k.replace("encoder.patch_embeddings.3.", "stages.3.patch_embeddings.")
        
        elif "encoder.block.0." in k: k = k.replace("encoder.block.0.", "stages.0.blocks.")
        elif "encoder.block.1." in k: k = k.replace("encoder.block.1.", "stages.1.blocks.")
        elif "encoder.block.2." in k: k = k.replace("encoder.block.2.", "stages.2.blocks.")
        elif "encoder.block.3." in k: k = k.replace("encoder.block.3.", "stages.3.blocks.")
        
        elif "encoder.layer_norm.0." in k: k = k.replace("encoder.layer_norm.0.", "stages.0.layer_norm.")
        elif "encoder.layer_norm.1." in k: k = k.replace("encoder.layer_norm.1.", "stages.1.layer_norm.")
        elif "encoder.layer_norm.2." in k: k = k.replace("encoder.layer_norm.2.", "stages.2.layer_norm.")
        elif "encoder.layer_norm.3." in k: k = k.replace("encoder.layer_norm.3.", "stages.3.layer_norm.")

        if "linear_c." in k:
            k = k.replace("linear_c.", "linear_projections.")
            
        k = k.replace("attention.self.query", "attention.q_proj")
        k = k.replace("attention.self.key", "attention.k_proj")
        k = k.replace("attention.self.value", "attention.v_proj")
        k = k.replace("attention.self.sr", "attention.sequence_reduction.sequence_reduction")
        k = k.replace("attention.self.layer_norm", "attention.sequence_reduction.layer_norm")
        k = k.replace("attention.output.dense", "attention.o_proj")
        
        k = k.replace("layer_norm_1", "layernorm_before")
        k = k.replace("layer_norm_2", "layernorm_after")
        
        k = k.replace("mlp.dense1", "mlp.fc1")
        k = k.replace("mlp.dense2", "mlp.fc2")
        
        new_sd[k] = v

    return new_sd


def migrate_segformer_keys_reverse(state_dict: dict) -> dict:
    """
    Migre les clés du nouveau format (utilisant `stages`)
    vers l'ancien format `transformers` (utilisant `encoder`).
    Utile si le checkpoint a été sauvegardé avec le nouveau format
    mais que le modèle attend l'ancien format.
    """
    # Vérification rapide : si l'ancien format est déjà là, on ne touche à rien
    if any(k.startswith("model.segformer.encoder.") for k in state_dict.keys()):
        return state_dict

    new_sd = {}
    for k, v in state_dict.items():
        if "stages.0.patch_embeddings." in k: k = k.replace("stages.0.patch_embeddings.", "encoder.patch_embeddings.0.")
        elif "stages.1.patch_embeddings." in k: k = k.replace("stages.1.patch_embeddings.", "encoder.patch_embeddings.1.")
        elif "stages.2.patch_embeddings." in k: k = k.replace("stages.2.patch_embeddings.", "encoder.patch_embeddings.2.")
        elif "stages.3.patch_embeddings." in k: k = k.replace("stages.3.patch_embeddings.", "encoder.patch_embeddings.3.")
        
        elif "stages.0.blocks." in k: k = k.replace("stages.0.blocks.", "encoder.block.0.")
        elif "stages.1.blocks." in k: k = k.replace("stages.1.blocks.", "encoder.block.1.")
        elif "stages.2.blocks." in k: k = k.replace("stages.2.blocks.", "encoder.block.2.")
        elif "stages.3.blocks." in k: k = k.replace("stages.3.blocks.", "encoder.block.3.")
        
        elif "stages.0.layer_norm." in k: k = k.replace("stages.0.layer_norm.", "encoder.layer_norm.0.")
        elif "stages.1.layer_norm." in k: k = k.replace("stages.1.layer_norm.", "encoder.layer_norm.1.")
        elif "stages.2.layer_norm." in k: k = k.replace("stages.2.layer_norm.", "encoder.layer_norm.2.")
        elif "stages.3.layer_norm." in k: k = k.replace("stages.3.layer_norm.", "encoder.layer_norm.3.")

        if "linear_projections." in k:
            k = k.replace("linear_projections.", "linear_c.")
            
        k = k.replace("attention.q_proj", "attention.self.query")
        k = k.replace("attention.k_proj", "attention.self.key")
        k = k.replace("attention.v_proj", "attention.self.value")
        k = k.replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
        k = k.replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
        k = k.replace("attention.o_proj", "attention.output.dense")
        
        k = k.replace("layernorm_before", "layer_norm_1")
        k = k.replace("layernorm_after", "layer_norm_2")
        
        k = k.replace("mlp.fc1", "mlp.dense1")
        k = k.replace("mlp.fc2", "mlp.dense2")
        
        new_sd[k] = v

    return new_sd


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions publiques
# ─────────────────────────────────────────────────────────────────────────────

def load_unet(
    checkpoint_path: Path | str = UNET_CKPT,
    device: torch.device = DEVICE,
) -> UNet:
    """
    Construit un UNet et charge les poids entraînés depuis le checkpoint.

    Args:
        checkpoint_path: Chemin vers unet_best.pth  (par défaut : outputs/models/unet_best.pth)
        device:          Périphérique de calcul (cpu / cuda)

    Returns:
        UNet en mode eval(), sur `device`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint U-Net introuvable : {checkpoint_path}\n"
            "Lancez d'abord l'entraînement pour générer outputs/models/unet_best.pth"
        )

    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(f"[load_models] U-Net chargé depuis : {checkpoint_path}")
    print(f"              Périphérique         : {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"              Paramètres           : {n_params:,}")
    return model


def load_segformer(
    checkpoint_path: Path | str = SEGFORMER_CKPT,
    device: torch.device = DEVICE,
) -> SegFormer:
    """
    Construit un SegFormer-B0 et charge les poids entraînés depuis le checkpoint.

    Args:
        checkpoint_path: Chemin vers segformer_best.pth  (par défaut : outputs/models/segformer_best.pth)
        device:          Périphérique de calcul (cpu / cuda)

    Returns:
        SegFormer en mode eval(), sur `device`.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint SegFormer introuvable : {checkpoint_path}\n"
            "Lancez d'abord l'entraînement pour générer outputs/models/segformer_best.pth"
        )

    model = SegFormer(num_classes=NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Migration des clés si nécessaire (anciennes versions de transformers)
    # Essayer d'abord la migration forward, puis reverse si ça échoue
    try:
        migrated_state = migrate_segformer_keys(state_dict)
        model.load_state_dict(migrated_state)
    except RuntimeError:
        # Si la migration forward échoue, essayer la reverse
        try:
            migrated_state = migrate_segformer_keys_reverse(state_dict)
            model.load_state_dict(migrated_state)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load SegFormer checkpoint with both forward and reverse key migration.\n"
                f"Checkpoint keys: {list(state_dict.keys())[:5]}...\n"
                f"Model keys: {list(model.state_dict().keys())[:5]}...\n"
                f"Error: {e}"
            )

    model.to(device)
    model.eval()

    print(f"[load_models] SegFormer chargé depuis : {checkpoint_path}")
    print(f"              Périphérique             : {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"              Paramètres               : {n_params:,}")
    return model


def load_both(
    unet_ckpt: Path | str = UNET_CKPT,
    segformer_ckpt: Path | str = SEGFORMER_CKPT,
    device: torch.device = DEVICE,
) -> dict:
    """
    Charge les deux modèles en une seule fois.

    Returns:
        {"unet": UNet, "segformer": SegFormer}  — tous deux sur `device` en eval().
    """
    print("=" * 60)
    print("Chargement des modèles depuis outputs/models/")
    print("=" * 60)
    unet      = load_unet(unet_ckpt, device)
    print()
    segformer = load_segformer(segformer_ckpt, device)
    print("=" * 60)
    return {"unet": unet, "segformer": segformer}


# ─────────────────────────────────────────────────────────────────────────────
# Test rapide en ligne de commande
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models = load_both(device=DEVICE)
    print("\nModèles disponibles :", list(models.keys()))
