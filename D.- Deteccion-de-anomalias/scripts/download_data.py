import argparse
import os
import sys


def download_dataset(target_dir: str) -> str:
    try:
        import kagglehub
    except ImportError:
        print("Instalando kagglehub...")
        os.system(f"{sys.executable} -m pip install kagglehub")
        import kagglehub

    print("Descargando dataset de fraudes con tarjetas de crédito...")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    src = os.path.join(path, "creditcard.csv")
    dst = os.path.join(target_dir, "creditcard.csv")
    import shutil
    shutil.copy2(src, dst)
    print(f"Dataset completo guardado en: {dst}")
    return dst


def create_sample(full_path: str, sample_path: str, n_normal: int = 2000, n_fraud: int = 50):
    import pandas as pd

    print(f"Creando muestra ({n_normal} normales + {n_fraud} fraudes)...")
    df = pd.read_csv(full_path)
    normal = df[df["Class"] == 0].sample(n=n_normal, random_state=42)
    fraud = df[df["Class"] == 1].sample(n=n_fraud, random_state=42)
    sample = pd.concat([normal, fraud]).sample(frac=1, random_state=42)
    sample.to_csv(sample_path, index=False)
    print(f"Muestra guardada en: {sample_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargar dataset de detección de fraudes")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directorio donde guardar los datos",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Solo generar la muestra a partir del dataset ya descargado",
    )
    parser.add_argument(
        "--n-normal",
        type=int,
        default=2000,
        help="Cantidad de transacciones normales en la muestra",
    )
    parser.add_argument(
        "--n-fraud",
        type=int,
        default=50,
        help="Cantidad de transacciones fraudulentas en la muestra",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    full_path = os.path.join(data_dir, "creditcard.csv")
    sample_path = os.path.join(data_dir, "creditcard_sample.csv")

    if args.sample_only:
        if not os.path.exists(full_path):
            print("El dataset completo no existe. Ejecutá sin --sample-only para descargarlo primero.")
            sys.exit(1)
        create_sample(full_path, sample_path, args.n_normal, args.n_fraud)
    else:
        download_dataset(data_dir)
        create_sample(full_path, sample_path, args.n_normal, args.n_fraud)
