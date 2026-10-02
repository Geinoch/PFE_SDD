from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CLASS_NAMES = (
    "Gliome",
    "Méningiome",
    "Tumeur pituitaire",
    "Aucune tumeur",
)
CLASS_ALIASES = {
    "gliome": 0,
    "glioma": 0,
    "gliomatumor": 0,
    "meningiome": 1,
    "meningioma": 1,
    "meningiomatumor": 1,
    "tumeurpituitaire": 2,
    "pituitarytumor": 2,
    "pituitary": 2,
    "aucunetumeur": 3,
    "notumor": 3,
    "healthy": 3,
}
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def normalize_name(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if character.isalnum()).lower()


def canonical_class(folder_name: str) -> int:
    key = normalize_name(folder_name)
    if key not in CLASS_ALIASES:
        raise ValueError(
            f"Classe inconnue: {folder_name!r}. Dossiers acceptés: "
            "gliome/glioma, meningiome/meningioma, tumeur pituitaire/pituitary, "
            "aucune tumeur/no tumor/healthy."
        )
    return CLASS_ALIASES[key]


class BrainMRIDataset(Dataset):
    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return image_tensor, label


class BrainCNN(nn.Module):
    def __init__(self, number_of_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    common = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ]
    training = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.RandomRotation(8)]
        + common[1:]
    )
    evaluation = transforms.Compose(common)
    return training, evaluation


def path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def load_patient_groups(manifest: Path, data_dir: Path) -> dict[str, str]:
    groups: dict[str, str] = {}
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"image", "patient_id"}.issubset(reader.fieldnames):
            raise ValueError("Le CSV doit contenir les colonnes 'image' et 'patient_id'.")
        for row in reader:
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                image_path = data_dir / image_path
            patient_id = row["patient_id"].strip()
            if not patient_id:
                raise ValueError(f"Identifiant patient vide pour {row['image']!r}.")
            key = path_key(image_path)
            if key in groups and groups[key] != patient_id:
                raise ValueError(f"Identifiants patients contradictoires pour {row['image']!r}.")
            groups[key] = patient_id
    return groups


def split_indices(
    labels: np.ndarray, groups: np.ndarray | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if groups is None:
        class_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        if np.any(class_counts < 5):
            raise ValueError("Il faut au moins 5 images par classe pour créer les partitions stratifiées.")
        train_validation, test = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.125,
            random_state=seed,
            stratify=labels[train_validation],
        )
        return train, validation, test

    for attempt in range(100):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        train_validation, test = next(outer.split(indices, labels, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + attempt)
        train_offset, validation_offset = next(
            inner.split(train_validation, labels[train_validation], groups[train_validation])
        )
        train = train_validation[train_offset]
        validation = train_validation[validation_offset]
        if all(set(labels[part]) == set(range(len(CLASS_NAMES))) for part in (train, validation, test)):
            return train, validation, test
    raise ValueError(
        "Impossible de placer les 4 classes dans train/validation/test sans séparer les patients. "
        "Vérifiez le CSV et le nombre de patients par classe."
    )


def build_samples(data_dir: Path) -> list[tuple[str, int]]:
    image_folder = datasets.ImageFolder(str(data_dir))
    folder_labels = {folder: canonical_class(folder) for folder in image_folder.classes}
    found_labels = set(folder_labels.values())
    expected_labels = set(range(len(CLASS_NAMES)))
    if found_labels != expected_labels:
        missing = [CLASS_NAMES[index] for index in sorted(expected_labels - found_labels)]
        raise ValueError(f"Il manque des dossiers de classes: {', '.join(missing)}")
    return [
        (image_path, folder_labels[image_folder.classes[folder_label]])
        for image_path, folder_label in image_folder.samples
    ]


def configure_spark_home() -> tuple[str | None, bool, str | None]:
    previous_home = os.environ.get("SPARK_HOME")
    if os.name != "nt" or previous_home:
        return previous_home, False, None

    import pyspark

    pyspark_home = Path(pyspark.__file__).resolve().parent
    spark_home = str(pyspark_home)
    temporary_drive = None
    if not spark_home.isascii():
        for letter in reversed("DEFGHIJKLMNOPQRSTUVWXYZ"):
            if not Path(f"{letter}:\\").exists():
                subprocess.run(
                    ["subst", f"{letter}:", spark_home],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                temporary_drive = letter
                spark_home = f"{letter}:\\"
                break
        if temporary_drive is None:
            raise RuntimeError(
                "Spark ne peut pas démarrer depuis un chemin Windows Unicode et aucun lecteur temporaire n'est disponible."
            )

    os.environ["SPARK_HOME"] = spark_home
    return previous_home, True, temporary_drive


def stage_hdfs_images(hdfs_uri: str, cache_dir: Path) -> int:
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError(
            "Le mode HDFS nécessite PySpark. Installez les dépendances de requirements.txt."
        ) from error

    parsed_uri = urlsplit(hdfs_uri)
    if parsed_uri.scheme != "hdfs" or not parsed_uri.netloc:
        raise ValueError("--hdfs-uri doit être une URI complète, par exemple hdfs://localhost:9000/PFE/Training/")

    previous_spark_home, changed_spark_home, temporary_drive = configure_spark_home()
    spark = None
    staged_count = 0
    try:
        spark = (
            SparkSession.builder.appName("PFE_MRI_CNN")
            .config("spark.hadoop.fs.defaultFS", f"{parsed_uri.scheme}://{parsed_uri.netloc}")
            .getOrCreate()
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        images = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .option("recursiveFileLookup", "true")
            .load(hdfs_uri)
            .select("path", "content")
        )
        for row in images.toLocalIterator():
            source_uri = row["path"]
            source_path = PurePosixPath(unquote(urlsplit(source_uri).path))
            class_index = canonical_class(source_path.parent.name)
            class_dir = cache_dir / CLASS_NAMES[class_index]
            class_dir.mkdir(parents=True, exist_ok=True)
            file_name = hashlib.sha256(source_uri.encode("utf-8")).hexdigest() + ".jpg"
            local_path = class_dir / file_name
            if not local_path.exists():
                local_path.write_bytes(bytes(row["content"]))
            staged_count += 1
    finally:
        try:
            if spark is not None:
                spark.stop()
        finally:
            if temporary_drive is not None:
                subprocess.run(
                    ["subst", f"{temporary_drive}:", "/d"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            if changed_spark_home:
                if previous_spark_home is None:
                    os.environ.pop("SPARK_HOME", None)
                else:
                    os.environ["SPARK_HOME"] = previous_spark_home

    if staged_count == 0:
        raise FileNotFoundError(f"Aucune image JPG trouvée dans {hdfs_uri}")
    print(f"Images HDFS mises en cache: {staged_count} dans {cache_dir}")
    return staged_count


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual: list[int] = []
    predicted: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
    return average_loss, macro_f1, actual, predicted


def predict_image(image_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = BrainCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    _, evaluation_transform = make_transforms(checkpoint["image_size"])

    with Image.open(image_path) as image:
        image_tensor = evaluation_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    print(f"Classe prédite: {CLASS_NAMES[predicted_index]}")
    print(f"Confiance du modèle: {probabilities[predicted_index].item():.1%}")
    print("Cette sortie est expérimentale et ne constitue pas un diagnostic médical.")


def train(args: argparse.Namespace) -> None:
    if args.hdfs_uri:
        data_dir = args.cache_dir.resolve()
        stage_hdfs_images(args.hdfs_uri, data_dir)
    else:
        data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dossier d'images introuvable: {data_dir}")

    samples = build_samples(data_dir)
    labels = np.asarray([label for _, label in samples], dtype=np.int64)
    groups: np.ndarray | None = None
    if args.groups_csv:
        group_map = load_patient_groups(args.groups_csv, data_dir)
        missing = [path for path, _ in samples if path_key(Path(path)) not in group_map]
        if missing:
            raise ValueError(f"Le CSV ne contient pas d'identifiant patient pour {len(missing)} image(s).")
        groups = np.asarray([group_map[path_key(Path(path))] for path, _ in samples])

    train_indices, validation_indices, test_indices = split_indices(labels, groups, args.seed)
    train_transform, evaluation_transform = make_transforms(args.image_size)
    train_data = BrainMRIDataset([samples[i] for i in train_indices], train_transform)
    validation_data = BrainMRIDataset([samples[i] for i in validation_indices], evaluation_transform)
    test_data = BrainMRIDataset([samples[i] for i in test_indices], evaluation_transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_validation_f1 = -1.0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Images: {len(samples)} | appareil: {device}")
    print(f"Partitions: train={len(train_data)}, validation={len(validation_data)}, test={len(test_data)}")
    if groups is None:
        print("Attention: partition par image. Fournissez --groups-csv si plusieurs images viennent d'un même patient.")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_f1, _, _ = run_epoch(
            model, validation_loader, criterion, device
        )
        print(
            f"Époque {epoch}/{args.epochs} - "
            f"perte train={train_loss:.4f}, F1 train={train_f1:.4f}, "
            f"perte validation={validation_loss:.4f}, F1 validation={validation_f1:.4f}"
        )
        if (validation_f1, -validation_loss) > (best_validation_f1, -best_validation_loss):
            best_validation_f1 = validation_f1
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Arrêt anticipé: la validation ne s'améliore plus.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_f1, actual, predicted = run_epoch(model, test_loader, criterion, device)
    report = classification_report(
        actual, predicted, labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES), output_dict=True, zero_division=0
    )
    results = {
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "classification_report": report,
        "best_validation_macro_f1": best_validation_f1,
        "class_names": CLASS_NAMES,
    }
    results_path = args.output_dir / "evaluation.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"F1 macro test: {test_f1:.4f} | perte test: {test_loss:.4f}")
    print(f"Modèle: {checkpoint_path}")
    print(f"Évaluation: {results_path}")
    print("Les résultats sont expérimentaux et ne constituent pas un diagnostic médical.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîner un CNN de classification d'IRM cérébrales.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, help="Dossier local contenant un sous-dossier par classe.")
    source.add_argument("--hdfs-uri", help="URI HDFS contenant un sous-dossier par classe.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/mri_training"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--groups-csv", type=Path, help="CSV facultatif avec les colonnes image,patient_id.")
    parser.add_argument("--predict-image", type=Path, help="Classer une image avec un modèle déjà entraîné.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.predict_image:
        if not args.predict_image.is_file():
            parser.error(f"Image introuvable: {args.predict_image}")
        if not args.model_path.is_file():
            parser.error(f"Modèle introuvable: {args.model_path}")
    else:
        if args.data_dir is None and args.hdfs_uri is None:
            parser.error("Indiquez --data-dir, --hdfs-uri ou --predict-image.")
        if args.image_size < 32 or args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
            parser.error("image-size doit être >= 32 et batch-size, epochs, patience doivent être positifs.")
        if args.learning_rate <= 0:
            parser.error("learning-rate doit être positif.")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.predict_image:
        predict_image(args.predict_image, args.model_path)
    else:
        train(args)


if __name__ == "__main__":
    main()
