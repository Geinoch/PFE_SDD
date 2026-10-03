import argparse
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, element_at, split

DEFAULT_INPUT_URI = "hdfs://localhost:9000/PFE/Training/"
DEFAULT_OUTPUT_URI = "hdfs://localhost:9000/PFE/Net_Training/"
EXPECTED_LABELS = {
    "Aucune tumeur",
    "Gliome",
    "Méningiome",
    "Tumeur pituitaire",
}


def validate_hdfs_uri(value: str, argument: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "hdfs" or not parsed.netloc:
        raise ValueError(f"{argument} doit être une URI HDFS complète: {value}")
    return value


def configure_spark_home() -> tuple[bool, str | None]:
    if os.name != "nt" or os.environ.get("SPARK_HOME"):
        return False, None

    import pyspark

    spark_home = str(Path(pyspark.__file__).resolve().parent)
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
            raise RuntimeError("Spark ne peut pas démarrer depuis un chemin Windows Unicode.")

    os.environ["SPARK_HOME"] = spark_home
    return True, temporary_drive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Préparer les images IRM dans HDFS au format Parquet.")
    parser.add_argument("--input-uri", default=DEFAULT_INPUT_URI, help="Dossier HDFS des images JPG.")
    parser.add_argument("--output-uri", default=DEFAULT_OUTPUT_URI, help="Destination Parquet dans HDFS.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remplacer explicitement une destination Parquet déjà existante.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_uri = validate_hdfs_uri(args.input_uri, "--input-uri")
    output_uri = validate_hdfs_uri(args.output_uri, "--output-uri")
    previous_spark_home = os.environ.get("SPARK_HOME")
    changed_spark_home = False
    temporary_drive = None
    spark = None

    try:
        changed_spark_home, temporary_drive = configure_spark_home()
        parsed_input_uri = urlsplit(input_uri)
        default_fs = f"{parsed_input_uri.scheme}://{parsed_input_uri.netloc}"
        spark = (
            SparkSession.builder.appName("PFE_MRI")
            .config("spark.hadoop.fs.defaultFS", default_fs)
            .getOrCreate()
        )

        images = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.jpg")
            .option("recursiveFileLookup", "true")
            .load(input_uri)
        )
        dataset = images.withColumn("label", element_at(split(col("path"), "/"), -2))
        class_counts = {row["label"]: row["count"] for row in dataset.groupBy("label").count().collect()}
        missing_labels = EXPECTED_LABELS - class_counts.keys()
        unknown_labels = class_counts.keys() - EXPECTED_LABELS
        if missing_labels or unknown_labels:
            raise ValueError(
                f"Classes HDFS invalides. Manquantes: {sorted(missing_labels)}; "
                f"inconnues: {sorted(unknown_labels)}"
            )

        print(f"Images chargées: {sum(class_counts.values())}")
        for label, count in sorted(class_counts.items()):
            print(f"{label}: {count}")

        write_mode = "overwrite" if args.overwrite else "errorifexists"
        dataset.write.mode(write_mode).parquet(output_uri)
        print(f"Dataset Parquet sauvegardé dans {output_uri}")
    finally:
        if spark is not None:
            spark.stop()
        if changed_spark_home:
            if previous_spark_home is None:
                os.environ.pop("SPARK_HOME", None)
            else:
                os.environ["SPARK_HOME"] = previous_spark_home
        if temporary_drive is not None:
            subprocess.run(
                ["subst", f"{temporary_drive}:", "/d"],
                check=False,
                capture_output=True,
                text=True,
            )


if __name__ == "__main__":
    main()
