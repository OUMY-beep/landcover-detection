"""
explore.py

Analyse exploratoire du dataset LoveDA.

Ce script permet de :
- Vérifier la structure du dataset
- Compter les images et les masques
- Vérifier la correspondance image ↔ masque
- Afficher les informations générales

Les analyses plus avancées (images, masques, classes,
graphiques...) seront ajoutées progressivement.
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import os


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "loveda"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
}


# ============================================================
# FONCTIONS D'AFFICHAGE
# ============================================================

def print_header(title: str):
    """Affiche un titre bien formaté."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def format_size(size_bytes: int) -> str:
    """
    Convertit une taille en octets vers une unité lisible.
    """

    units = ["B", "KB", "MB", "GB", "TB"]

    size = float(size_bytes)

    for unit in units:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


# ============================================================
# STRUCTURE DU DATASET
# ============================================================

def print_dataset_location():
    """Affiche le chemin du dataset."""

    print_header("LOCALISATION DU DATASET")

    if DATASET_ROOT.exists():

        print(DATASET_ROOT.resolve())

    else:

        print("❌ Dataset introuvable.")


def compute_dataset_size():
    """
    Calcule la taille totale du dataset.
    """

    total_size = 0

    for file in DATASET_ROOT.rglob("*"):

        if file.is_file():

            total_size += file.stat().st_size

    return total_size


def display_folder_tree():
    """
    Affiche l'arborescence du dataset.
    """

    print_header("STRUCTURE DU DATASET")

    for path in sorted(DATASET_ROOT.rglob("*")):

        level = len(path.relative_to(DATASET_ROOT).parts)

        indent = "    " * (level - 1)

        if path.is_dir():

            print(f"{indent}📁 {path.name}")


def analyze_folders():
    """
    Analyse tous les dossiers contenant des images.
    """

    print_header("ANALYSE DES DOSSIERS")

    folder_statistics = defaultdict(int)

    total_images = 0
    total_masks = 0

    for folder in sorted(DATASET_ROOT.rglob("*")):

        if not folder.is_dir():
            continue

        files = [
            f
            for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(files) == 0:
            continue

        relative = folder.relative_to(DATASET_ROOT)

        folder_statistics[str(relative)] = len(files)

        print(f"\n📂 {relative}")
        print(f"Nombre de fichiers : {len(files)}")

        folder_name = folder.name.lower()

        if "image" in folder_name:

            total_images += len(files)

        elif "mask" in folder_name:

            total_masks += len(files)

    return folder_statistics, total_images, total_masks


def verify_image_mask_pairs():
    """
    Vérifie que chaque image possède un masque.
    """

    print_header("VÉRIFICATION IMAGE ↔ MASQUE")

    image_folder = None
    mask_folder = None

    for folder in DATASET_ROOT.rglob("*"):

        if not folder.is_dir():
            continue

        name = folder.name.lower()

        if "image" in name:

            image_folder = folder

        elif "mask" in name:

            mask_folder = folder

    if image_folder is None:

        print("❌ Dossier images introuvable.")
        return

    if mask_folder is None:

        print("❌ Dossier masks introuvable.")
        return

    images = {
        file.stem
        for file in image_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    }

    masks = {
        file.stem
        for file in mask_folder.iterdir()
        if file.suffix.lower() in IMAGE_EXTENSIONS
    }

    missing_masks = images - masks

    missing_images = masks - images

    if len(missing_masks) == 0 and len(missing_images) == 0:

        print("✅ Toutes les images possèdent un masque.")

    else:

        print(f"Images sans masque : {len(missing_masks)}")
        print(f"Masques sans image : {len(missing_images)}")


def display_summary(total_images: int,
                    total_masks: int):
    """
    Affiche un résumé général.
    """

    print_header("RÉSUMÉ")

    print(f"Nombre total d'images : {total_images}")
    print(f"Nombre total de masques : {total_masks}")

    print(f"Taille du dataset : {format_size(compute_dataset_size())}")

# ============================================================
# ANALYSE DES IMAGES
# ============================================================

import numpy as np
from PIL import Image


def analyze_images():
    """
    Analyse toutes les images du dataset.
    """

    print_header("ANALYSE DES IMAGES")

    image_folder = None

    for folder in DATASET_ROOT.rglob("*"):

        if folder.is_dir() and "image" in folder.name.lower():
            image_folder = folder
            break

    if image_folder is None:
        print("❌ Dossier images introuvable.")
        return

    image_files = sorted([
        f for f in image_folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    widths = []
    heights = []
    file_sizes = []

    modes = defaultdict(int)
    formats = defaultdict(int)
    dtypes = defaultdict(int)

    corrupted = []

    print(f"Analyse de {len(image_files)} images...")

    for file in tqdm(image_files):

        try:

            img = Image.open(file)

            widths.append(img.width)
            heights.append(img.height)

            modes[img.mode] += 1

            if img.format:
                formats[img.format] += 1

            array = np.array(img)

            dtypes[str(array.dtype)] += 1

            file_sizes.append(file.stat().st_size)

        except Exception:

            corrupted.append(file.name)

    print()

    print(f"Nombre d'images : {len(image_files)}")

    print("\nRésolution :")

    print(f"   Largeur min : {min(widths)}")
    print(f"   Largeur max : {max(widths)}")
    print(f"   Largeur moyenne : {np.mean(widths):.2f}")

    print()

    print(f"   Hauteur min : {min(heights)}")
    print(f"   Hauteur max : {max(heights)}")
    print(f"   Hauteur moyenne : {np.mean(heights):.2f}")

    print()

    print("Modes rencontrés :")

    for mode, count in modes.items():
        print(f"   {mode} : {count}")

    print()

    print("Formats rencontrés :")

    for fmt, count in formats.items():
        print(f"   {fmt} : {count}")

    print()

    print("Types de données :")

    for dtype, count in dtypes.items():
        print(f"   {dtype} : {count}")

    print()

    print(f"Taille moyenne des fichiers : {format_size(int(np.mean(file_sizes)))}")

    if corrupted:

        print()

        print("Images corrompues :")

        for name in corrupted:
            print(name)

    else:

        print("✅ Aucune image corrompue.")

# ============================================================
# ANALYSE DES MASQUES
# ============================================================

def analyze_masks():
    """
    Analyse tous les masques.
    """

    print_header("ANALYSE DES MASQUES")

    mask_folder = None

    for folder in DATASET_ROOT.rglob("*"):

        if folder.is_dir() and "mask" in folder.name.lower():
            mask_folder = folder
            break

    if mask_folder is None:
        print("❌ Dossier masks introuvable.")
        return

    mask_files = sorted([
        f for f in mask_folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    widths = []
    heights = []

    modes = defaultdict(int)
    formats = defaultdict(int)
    dtypes = defaultdict(int)

    unique_values = set()

    corrupted = []

    print(f"Analyse de {len(mask_files)} masques...")

    for file in tqdm(mask_files):

        try:

            mask = Image.open(file)

            widths.append(mask.width)
            heights.append(mask.height)

            modes[mask.mode] += 1

            if mask.format:
                formats[mask.format] += 1

            array = np.array(mask)

            dtypes[str(array.dtype)] += 1

            unique_values.update(np.unique(array).tolist())

        except Exception:

            corrupted.append(file.name)

    print()

    print(f"Nombre de masques : {len(mask_files)}")

    print()

    print("Résolution :")

    print(f"   Largeur min : {min(widths)}")
    print(f"   Largeur max : {max(widths)}")
    print(f"   Largeur moyenne : {np.mean(widths):.2f}")

    print()

    print(f"   Hauteur min : {min(heights)}")
    print(f"   Hauteur max : {max(heights)}")
    print(f"   Hauteur moyenne : {np.mean(heights):.2f}")

    print()

    print("Modes rencontrés :")

    for mode, count in modes.items():
        print(f"   {mode} : {count}")

    print()

    print("Formats rencontrés :")

    for fmt, count in formats.items():
        print(f"   {fmt} : {count}")

    print()

    print("Types de données :")

    for dtype, count in dtypes.items():
        print(f"   {dtype} : {count}")

    print()

    print("Valeurs présentes dans les masques :")

    print(sorted(unique_values))

    print()

    print(f"Nombre total de classes détectées : {len(unique_values)}")

    if corrupted:

        print()

        print("Masques corrompus :")

        for name in corrupted:
            print(name)

    else:

        print("✅ Aucun masque corrompu.")

FAST_MODE = True
SAMPLE_SIZE = 200

# ============================================================
# DISTRIBUTION DES CLASSES
# ============================================================

from collections import Counter
import random


CLASS_NAMES = {
    0: "Background",
    1: "Building",
    2: "Road",
    3: "Water",
    4: "Barren",
    5: "Forest",
    6: "Agricultural"
}


def analyze_class_distribution():

    print_header("DISTRIBUTION DES CLASSES")

    mask_folder = None

    for folder in DATASET_ROOT.rglob("*"):
        if folder.is_dir() and "mask" in folder.name.lower():
            mask_folder = folder
            break

    if mask_folder is None:
        print("❌ Dossier des masques introuvable.")
        return None

    mask_files = sorted([
        f for f in mask_folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if FAST_MODE:
        random.seed(42)
        mask_files = random.sample(
            mask_files,
            min(SAMPLE_SIZE, len(mask_files))
        )

        print(f"⚡ Mode FAST : {len(mask_files)} masques analysés")

    else:

        print(f"🐢 Mode FULL : {len(mask_files)} masques analysés")

    counter = Counter()

    total_pixels = 0

    for file in tqdm(mask_files):

        mask = np.array(Image.open(file))

        values, counts = np.unique(mask, return_counts=True)

        for value, count in zip(values, counts):

            counter[int(value)] += int(count)

            total_pixels += int(count)

    print()

    print("Distribution des classes :\n")

    distribution = {}

    for cls in sorted(counter.keys()):

        percentage = counter[cls] * 100 / total_pixels

        distribution[cls] = percentage

        name = CLASS_NAMES.get(cls, f"Classe {cls}")

        print(f"{name:<15} : {percentage:6.2f}%")

    dominant = max(counter, key=counter.get)
    rarest = min(counter, key=counter.get)

    print()

    print(f"Classe dominante : {CLASS_NAMES.get(dominant)}")

    print(f"Classe la moins représentée : {CLASS_NAMES.get(rarest)}")

    return distribution

# ============================================================
# GRAPHIQUES
# ============================================================

import matplotlib.pyplot as plt


OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(distribution):

    if distribution is None:
        return

    names = [
        CLASS_NAMES.get(k, str(k))
        for k in distribution.keys()
    ]

    values = list(distribution.values())

    # Histogramme

    plt.figure(figsize=(10,5))

    plt.bar(names, values)

    plt.title("Distribution des classes")

    plt.ylabel("Pourcentage (%)")

    plt.xticks(rotation=25)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / "class_distribution_bar.png",
        dpi=300
    )

    plt.close()

    # Camembert

    plt.figure(figsize=(8,8))

    plt.pie(
        values,
        labels=names,
        autopct="%1.1f%%"
    )

    plt.title("Répartition des classes")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURES / "class_distribution_pie.png",
        dpi=300
    )

    plt.close()

    print()

    print("✅ Graphiques enregistrés dans :")

    print(OUTPUT_FIGURES)


# ============================================================
# GÉNÉRATION DU RAPPORT
# ============================================================

from datetime import datetime

OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def generate_report(total_images,
                    total_masks,
                    distribution):

    report_path = OUTPUT_REPORTS / "dataset_report.txt"

    with open(report_path, "w", encoding="utf-8") as report:

        report.write("=" * 70 + "\n")
        report.write("LOVEDA DATASET REPORT\n")
        report.write("=" * 70 + "\n\n")

        report.write(f"Date : {datetime.now()}\n\n")

        report.write(f"Dataset : {DATASET_ROOT}\n\n")

        report.write(f"Nombre d'images : {total_images}\n")
        report.write(f"Nombre de masques : {total_masks}\n\n")

        report.write(f"Taille totale : {format_size(compute_dataset_size())}\n\n")

        report.write("Distribution des classes\n")
        report.write("-" * 40 + "\n")

        if distribution is not None:

            for cls, percentage in distribution.items():

                name = CLASS_NAMES.get(cls, str(cls))

                report.write(
                    f"{name:<20} {percentage:.2f}%\n"
                )

        report.write("\n")

        report.write("Analyse terminée avec succès.\n")

    print()

    print("✅ Rapport enregistré :")

    print(report_path)

# ============================================================
# SAUVEGARDE D'EXEMPLES
# ============================================================

import random
import matplotlib.pyplot as plt

OUTPUT_EXAMPLES = OUTPUT_FIGURES / "examples"
OUTPUT_EXAMPLES.mkdir(parents=True, exist_ok=True)


def save_random_examples(n_examples=5):

    print_header("SAUVEGARDE D'EXEMPLES")

    image_folder = None
    mask_folder = None

    for folder in DATASET_ROOT.rglob("*"):

        if folder.is_dir():

            if "image" in folder.name.lower():

                image_folder = folder

            elif "mask" in folder.name.lower():

                mask_folder = folder

    if image_folder is None or mask_folder is None:

        print("❌ Impossible de trouver les dossiers.")
        return

    image_files = sorted([
        f for f in image_folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ])

    random.seed(42)

    examples = random.sample(
        image_files,
        min(n_examples, len(image_files))
    )

    for image_path in examples:

        mask_path = mask_folder / image_path.name

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(mask)
        plt.title("Masque")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_EXAMPLES /
            f"{image_path.stem}.png",
            dpi=300
        )

        plt.close()

    print(f"{len(examples)} exemples sauvegardés.")



# ============================================================
# RÉSUMÉ FINAL
# ============================================================

def final_summary():

    print_header("RÉSUMÉ FINAL")

    print("✅ Exploration terminée avec succès.\n")

    print("Résultats générés :")

    print()

    print("📊 outputs/figures/")

    print("   • class_distribution_bar.png")

    print("   • class_distribution_pie.png")

    print("   • examples/")

    print()

    print("📄 outputs/reports/")

    print("   • dataset_report.txt")

    print()

    print("Le dataset est maintenant prêt")

    print("pour la création du DataLoader.")



if __name__ == "__main__":

    print_dataset_location()

    display_folder_tree()

    _, total_images, total_masks = analyze_folders()

    display_summary(
        total_images,
        total_masks
    )

    verify_image_mask_pairs()

    analyze_images()

    analyze_masks()

    distribution = analyze_class_distribution()

    plot_class_distribution(distribution)

    save_random_examples()

    generate_report(
        total_images,
        total_masks,
        distribution
    )

    final_summary()