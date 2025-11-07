# EDA: limpeza, feature engineering, análises univariadas/multivariadas e seleção de atributos
import os
import sys
from pathlib import Path
from typing import List, Dict
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # salvar figuras em arquivo
import matplotlib.pyplot as plt
import seaborn as sns

from eda_utils import list_images, is_image_corrupted, compute_phash, compute_features, ensure_dir, CLASSES

def scan_dataset(dataset_dir: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for label, path in list_images(dataset_dir):
        rows.append({"label": label, "path": path})
    df = pd.DataFrame(rows)
    return df


def clean_and_features(df: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    ensure_dir(out_dir)
    # checa corrompidos e phash
    corrupted_flags = []
    phashes = []
    for p in df["path"]:
        corrupted_flags.append(is_image_corrupted(p))
        phashes.append(compute_phash(p))
    df = df.copy()
    df["is_corrupted"] = corrupted_flags
    df["phash"] = phashes

    # engenheira de atributos
    feat_rows: List[Dict] = []
    for p in df["path"]:
        feat_rows.append(compute_features(p))
    feat_df = pd.DataFrame(feat_rows)
    full = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    full.to_csv(Path(out_dir)/"image_features.csv", index=False)
    return full


def find_duplicates(df: pd.DataFrame, max_hamming: int = 3) -> pd.DataFrame:
    # marca prováveis duplicatas usando distância de Hamming de phash
    try:
        import imagehash
    except Exception:
        df["dup_group"] = None
        return df

    df = df.copy()
    hashes = df["phash"].fillna("").tolist()
    groups = [-1] * len(hashes)
    group_id = 0
    for i in range(len(hashes)):
        if groups[i] != -1 or not hashes[i]:
            continue
        groups[i] = group_id
        for j in range(i+1, len(hashes)):
            if groups[j] == -1 and hashes[j]:
                d = imagehash.hex_to_hash(hashes[i]) - imagehash.hex_to_hash(hashes[j])
                if d <= max_hamming:
                    groups[j] = group_id
        group_id += 1
    df["dup_group"] = groups
    return df


def plots_univariate(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    # distribuição de classes
    plt.figure(figsize=(8,4))
    sns.countplot(x="label", data=df, order=CLASSES)
    plt.title("Distribuição de classes")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"class_distribution.png", dpi=150)
    plt.close()

    # histogramas por feature
    features = [
        "brightness_mean","sharpness_lapl_var","colorfulness","entropy_gray",
        "width","height","aspect_ratio"
    ]
    for feat in features:
        if feat not in df.columns:
            continue
        plt.figure(figsize=(8,4))
        has_any = False
        for cls in CLASSES:
            serie = df[df.label==cls][feat].dropna()
            # evita warnings quando não há variância ou dados insuficientes
            if serie.size < 5 or serie.nunique() <= 1:
                continue
            sns.kdeplot(serie, label=cls, common_norm=False, warn_singular=False)
            has_any = True
        plt.title(f"Distribuição: {feat}")
        if has_any:
            plt.legend()
        plt.tight_layout()
        plt.savefig(Path(out_dir)/f"kde_{feat}.png", dpi=150)
        plt.close()


def plots_correlation(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        return
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Matriz de correlação (numéricas)")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"correlation_heatmap.png", dpi=150)
    plt.close()


def plots_pca(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    base_feats = [
        "brightness_mean","sharpness_lapl_var","colorfulness","entropy_gray",
        "mean_r","mean_g","mean_b","std_r","std_g","std_b","aspect_ratio","area_px"
    ]
    feats = [c for c in base_feats if c in df.columns]
    if len(feats) < 2:
        return
    X = df[feats].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    y = df["label"].values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    pc_df = pd.DataFrame({"pc1": Xp[:,0], "pc2": Xp[:,1], "label": y})
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=pc_df, x="pc1", y="pc2", hue="label", hue_order=CLASSES, s=25)
    plt.title("PCA (2D) dos atributos engenheirados")
    plt.tight_layout()
    plt.savefig(Path(out_dir)/"pca_scatter.png", dpi=150)
    plt.close()


def feature_selection(df: pd.DataFrame, out_dir: str) -> None:
    ensure_dir(out_dir)
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression

    base_feats = [
        "brightness_mean","sharpness_lapl_var","colorfulness","entropy_gray",
        "mean_r","mean_g","mean_b","std_r","std_g","std_b","aspect_ratio","area_px"
    ]
    feats = [c for c in base_feats if c in df.columns]
    if len(feats) == 0:
        return
    X = df[feats].replace([np.inf,-np.inf], np.nan).fillna(0.0).values
    y = df["label"].values
    # codifica labels
    labels = {c:i for i,c in enumerate(CLASSES)}
    y_enc = np.array([labels.get(v, -1) for v in y])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Mutual Information
    mi = mutual_info_classif(Xs, y_enc, random_state=42, discrete_features=False)
    mi_series = pd.Series(mi, index=feats).sort_values(ascending=False)
    mi_series.to_csv(Path(out_dir)/"feature_importance_mutual_info.csv")

    # RandomForest importance
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xs, y_enc)
    rf_imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False)
    rf_imp.to_csv(Path(out_dir)/"feature_importance_random_forest.csv")

    # Logistic L1 (só para ranking adicional)
    try:
        lr = LogisticRegression(penalty="l1", solver="saga", max_iter=2000, multi_class='multinomial')
        lr.fit(Xs, y_enc)
        lr_imp = pd.Series(np.mean(np.abs(lr.coef_), axis=0), index=feats).sort_values(ascending=False)
        lr_imp.to_csv(Path(out_dir)/"feature_importance_logreg_l1.csv")
    except Exception:
        pass


def main(dataset_dir: str, out_dir: str) -> None:
    ensure_dir(out_dir)
    print("[EDA] Escaneando dataset...", dataset_dir)
    df = scan_dataset(dataset_dir)
    df.to_csv(Path(out_dir)/"files_list.csv", index=False)
    print(f"[EDA] {len(df)} imagens encontradas")

    print("[EDA] Limpando e gerando features...")
    full = clean_and_features(df, out_dir)
    full = find_duplicates(full)
    full.to_csv(Path(out_dir)/"image_features_with_meta.csv", index=False)

    # salva resumo de limpeza
    cleaning = {
        "num_images": int(len(full)),
        "num_corrupted": int(full["is_corrupted"].sum()),
        "num_with_phash": int(full["phash"].notna().sum())
    }
    Path(out_dir, "summary.json").write_text(json.dumps(cleaning, indent=2))

    print("[EDA] Gerando gráficos univariados...")
    plots_univariate(full, Path(out_dir)/"plots_univariate")

    print("[EDA] Gerando correlação e PCA...")
    plots_correlation(full, Path(out_dir)/"plots_multivariate")
    plots_pca(full, Path(out_dir)/"plots_multivariate")

    print("[EDA] Seleção de atributos...")
    feature_selection(full, Path(out_dir)/"feature_selection")

    print("[EDA] Concluído. Resultados em:", out_dir)


if __name__ == "__main__":
    # Uso: python eda.py [dataset_dir] [out_dir]
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path("..")/"data"/"dataset-resized")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path(".")/"outputs")
    main(dataset_dir, out_dir)
