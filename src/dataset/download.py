"""
download.py

Téléchargement automatique du dataset LoveDA via KaggleHub.
Le téléchargement n'est effectué que si le dataset n'est pas déjà présent.
"""

from pathlib import Path
import shutil
import kagglehub


class LoveDADownloader:
    """
    Télécharge automatiquement LoveDA via KaggleHub
    puis le copie dans data/raw/loveda/.
    """

    def __init__(self):
        self.dataset_name = "growinfame/loveda"
        PROJECT_ROOT = Path(__file__).resolve().parents[2]

        self.destination = PROJECT_ROOT / "data" / "raw" / "loveda"

    def dataset_exists(self) -> bool:
        """
        Vérifie si le dataset existe déjà.
        """
        return self.destination.exists() and any(self.destination.iterdir())

    def download(self):
        """
        Télécharge le dataset uniquement s'il n'existe pas.
        """

        if self.dataset_exists():
            print("✅ LoveDA est déjà présent.")
            print(f"📂 {self.destination.resolve()}")
            return self.destination

        print("⬇️ Téléchargement de LoveDA...")

        downloaded_path = Path(
            kagglehub.dataset_download(self.dataset_name)
        )

        print("✅ Téléchargement terminé.")

        print("📦 Copie des fichiers...")

        shutil.copytree(
            downloaded_path,
            self.destination,
            dirs_exist_ok=True
        )

        print("✅ Dataset disponible dans :")
        print(self.destination.resolve())

        return self.destination


if __name__ == "__main__":
    downloader = LoveDADownloader()
    downloader.download()