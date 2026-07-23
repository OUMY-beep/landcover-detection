"""
benchmark.py
============
Benchmark de performance pour U-Net et SegFormer.

Mesure :
    - Temps moyen d'inférence (ms/image, sur N warmup + K runs)
    - FPS  (images par seconde)
    - Mémoire GPU utilisée  (MB, si CUDA disponible)
    - Mémoire CPU  (MB, via psutil)
    - Taille du modèle  (MB, paramètres + buffers)
    - Nombre de paramètres

Sorties :
    - Rapport console
    - outputs/reports/benchmark.json
    - outputs/reports/benchmark_chart.png  (graphique comparatif)

Utilisation :
    python src/evaluation/benchmark.py
    python src/evaluation/benchmark.py --model unet
    python src/evaluation/benchmark.py --batch-size 4 --runs 100
    python src/evaluation/benchmark.py --no-warmup
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# ── Chemins ───────────────────────────────────────────────────────────────────
SRC_ROOT     = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Imports internes ──────────────────────────────────────────────────────────
from evaluation.load_models import DEVICE, load_segformer, load_unet
from preprocessing.transforms import NUM_CLASSES

# ── psutil (mémoire CPU) ─────────────────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── Image size par défaut (même que l'entraînement) ──────────────────────────
IMAGE_H, IMAGE_W = 512, 512


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model: nn.Module) -> float:
    """Taille des paramètres + buffers en MB."""
    total = sum(
        p.nelement() * p.element_size()
        for p in list(model.parameters()) + list(model.buffers())
    )
    return total / (1024 ** 2)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_cpu_memory_mb() -> float:
    """Retourne la mémoire RAM du processus courant en MB."""
    if not _HAS_PSUTIL:
        return -1.0
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 ** 2)


def get_gpu_memory_mb(device: torch.device) -> float:
    """Retourne la mémoire GPU allouée en MB (0 si CPU)."""
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark principal
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_model(
    model:      nn.Module,
    model_name: str,
    device:     torch.device,
    batch_size: int   = 1,
    warmup_runs: int  = 10,
    timed_runs: int   = 100,
    img_h:      int   = IMAGE_H,
    img_w:      int   = IMAGE_W,
) -> dict:
    """
    Mesure les performances du modèle sur des images synthétiques.

    Args:
        model:       Modèle à benchmarker.
        model_name:  Nom du modèle (pour les logs).
        device:      Périphérique cible.
        batch_size:  Taille du batch (défaut : 1, réaliste pour la démo/prod).
        warmup_runs: Nombre de passes d'échauffement (ignorées dans les stats).
        timed_runs:  Nombre de passes chronométrées.
        img_h/w:     Dimensions des images d'entrée.

    Returns:
        dict avec avg_ms, std_ms, min_ms, max_ms, fps, gpu_memory_mb,
              cpu_memory_mb, model_size_mb, num_parameters.
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, img_h, img_w, device=device)

    print(f"\n  Benchmarking {model_name}  "
          f"(device={device}, batch={batch_size}, runs={warmup_runs}+{timed_runs})")

    # ── Échauffement ──────────────────────────────────────────────────────────
    print(f"  Warmup ({warmup_runs} passes)…", end=" ", flush=True)
    for _ in range(warmup_runs):
        _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    print("OK")

    # ── Mémoire GPU avant les passes chronométrées ────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    cpu_mem_before = get_cpu_memory_mb()

    # ── Passes chronométrées ─────────────────────────────────────────────────
    latencies_ms: list[float] = []

    for _ in range(timed_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _  = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    # ── Statistiques ─────────────────────────────────────────────────────────
    lat = np.array(latencies_ms)
    avg_ms = float(np.mean(lat))
    std_ms = float(np.std(lat))
    min_ms = float(np.min(lat))
    max_ms = float(np.max(lat))
    p95_ms = float(np.percentile(lat, 95))

    # FPS : images par seconde (batch_size images traitées en avg_ms ms)
    fps = batch_size * 1000.0 / avg_ms if avg_ms > 0 else 0.0

    # Mémoire
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_mem_mb = 0.0

    cpu_mem_mb = max(get_cpu_memory_mb() - cpu_mem_before, 0.0)

    results = {
        "model_name"      : model_name,
        "device"          : str(device),
        "batch_size"      : batch_size,
        "warmup_runs"     : warmup_runs,
        "timed_runs"      : timed_runs,
        "avg_ms_per_batch": round(avg_ms,  2),
        "avg_ms_per_image": round(avg_ms / batch_size, 2),
        "std_ms"          : round(std_ms,  2),
        "min_ms"          : round(min_ms,  2),
        "max_ms"          : round(max_ms,  2),
        "p95_ms"          : round(p95_ms,  2),
        "fps"             : round(fps,     1),
        "gpu_memory_mb"   : round(gpu_mem_mb, 1),
        "cpu_memory_mb"   : round(cpu_mem_mb, 1),
        "model_size_mb"   : round(model_size_mb(model), 2),
        "num_parameters"  : count_parameters(model),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Affichage console
# ─────────────────────────────────────────────────────────────────────────────

def print_benchmark_report(results: dict) -> None:
    sep  = "=" * 65
    sep2 = "-" * 65
    print(f"\n{sep}")
    print(f"  BENCHMARK — {results['model_name'].upper()}")
    print(sep)
    print(f"  Périphérique          : {results['device']}")
    print(f"  Taille du batch       : {results['batch_size']}")
    print(f"  Passes chronométrées  : {results['timed_runs']}")
    print(sep2)
    print(f"  Latence moy./batch    : {results['avg_ms_per_batch']:>8.2f} ms")
    print(f"  Latence moy./image    : {results['avg_ms_per_image']:>8.2f} ms")
    print(f"  Latence std           : {results['std_ms']:>8.2f} ms")
    print(f"  Latence min           : {results['min_ms']:>8.2f} ms")
    print(f"  Latence max           : {results['max_ms']:>8.2f} ms")
    print(f"  Latence p95           : {results['p95_ms']:>8.2f} ms")
    print(f"  FPS                   : {results['fps']:>8.1f} img/s")
    print(sep2)
    print(f"  Mémoire GPU           : {results['gpu_memory_mb']:>8.1f} MB")
    print(f"  Mémoire CPU (delta)   : {results['cpu_memory_mb']:>8.1f} MB")
    print(f"  Taille du modèle      : {results['model_size_mb']:>8.2f} MB")
    print(f"  Nombre de paramètres  : {results['num_parameters']:>14,}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Graphique comparatif
# ─────────────────────────────────────────────────────────────────────────────

def save_benchmark_chart(results_list: list[dict]) -> None:
    """
    Génère un graphique à barres comparant les deux modèles sur plusieurs
    métriques clés, et le sauvegarde dans outputs/reports/benchmark_chart.png.
    """
    if len(results_list) < 2:
        return

    r0, r1   = results_list[0], results_list[1]
    names    = [r["model_name"] for r in results_list]
    colors   = ["#58a6ff", "#3fb950"]   # bleu U-Net, vert SegFormer
    bg_color = "#0d1117"
    ax_color = "#161b22"
    txt_clr  = "#e6edf3"
    grid_clr = "#30363d"

    metrics = [
        ("Latence moy./image (ms)", "avg_ms_per_image",   False),
        ("FPS",                     "fps",                 True),
        ("Mémoire GPU (MB)",        "gpu_memory_mb",       False),
        ("Taille modèle (MB)",      "model_size_mb",       False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    fig.patch.set_facecolor(bg_color)

    for ax, (title, key, higher_better) in zip(axes, metrics):
        vals = [r.get(key, 0) for r in results_list]

        bars = ax.bar(names, vals, color=colors, width=0.5,
                      edgecolor="#30363d", linewidth=0.8)

        # Annotations
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.1f}",
                ha="center", va="bottom",
                color=txt_clr, fontsize=10, fontweight="bold",
            )

        ax.set_facecolor(ax_color)
        ax.set_title(title, color=txt_clr, fontsize=11, pad=10)
        ax.tick_params(colors=txt_clr, labelsize=9)
        ax.yaxis.set_tick_params(colors=txt_clr)
        ax.xaxis.set_tick_params(colors=txt_clr)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_clr)
        ax.yaxis.grid(True, color=grid_clr, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        # Indication de la direction souhaitée
        note = "↑ mieux" if higher_better else "↓ mieux"
        ax.text(0.97, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=SUBTITLE_CLR,
                fontstyle="italic")

    fig.suptitle(
        "Benchmark : U-Net vs SegFormer",
        color=txt_clr, fontsize=14, y=1.02,
    )
    plt.tight_layout()

    out = REPORTS_DIR / "benchmark_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=bg_color)
    plt.close(fig)
    print(f"  → Graphique sauvegardé : {out}")


# Couleur texte secondaire (défini ici pour la référence dans save_benchmark_chart)
SUBTITLE_CLR = "#8b949e"


# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_benchmark_json(results_list: list[dict]) -> None:
    out = REPORTS_DIR / "benchmark.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"  → Benchmark JSON : {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de performance : U-Net vs SegFormer"
    )
    parser.add_argument(
        "--model", choices=["unet", "segformer", "both"], default="both",
        help="Modèle(s) à benchmarker (défaut : both)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Taille du batch d'entrée (défaut : 1)",
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="Nombre de passes chronométrées (défaut : 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Nombre de passes d'échauffement (défaut : 10)",
    )
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Désactiver le warmup (non recommandé sur GPU)",
    )
    args = parser.parse_args()

    warmup = 0 if args.no_warmup else args.warmup

    print(f"\n{'='*65}")
    print("  BENCHMARK DE PERFORMANCE")
    print(f"{'='*65}")
    print(f"  Périphérique  : {DEVICE}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  Warmup        : {warmup} passes")
    print(f"  Runs          : {args.runs} passes")

    if not _HAS_PSUTIL:
        print("\n  [INFO] psutil non installé → mémoire CPU non disponible.")
        print("         pip install psutil")

    results_list: list[dict] = []

    # ── U-Net ─────────────────────────────────────────────────────────────────
    if args.model in ("unet", "both"):
        unet = load_unet(device=DEVICE)
        res  = benchmark_model(
            unet, "U-Net", DEVICE,
            batch_size=args.batch_size,
            warmup_runs=warmup,
            timed_runs=args.runs,
        )
        print_benchmark_report(res)
        results_list.append(res)
        del unet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── SegFormer ─────────────────────────────────────────────────────────────
    if args.model in ("segformer", "both"):
        segformer = load_segformer(device=DEVICE)
        res       = benchmark_model(
            segformer, "SegFormer", DEVICE,
            batch_size=args.batch_size,
            warmup_runs=warmup,
            timed_runs=args.runs,
        )
        print_benchmark_report(res)
        results_list.append(res)
        del segformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    if results_list:
        save_benchmark_json(results_list)
    if len(results_list) == 2:
        save_benchmark_chart(results_list)

    print("\nBenchmark terminé.\n")


if __name__ == "__main__":
    main()
