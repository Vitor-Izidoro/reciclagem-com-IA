# Report Builder: gera um relatório HTML único a partir das saídas da EDA
import sys
import json
from pathlib import Path
import pandas as pd


def img_tag(path: Path, width: int = 640) -> str:
    if not path.exists():
        return f"<p style='color:#a00'>[imagem não encontrada: {path.as_posix()}]</p>"
    return f"<img src='{path.as_posix()}' width='{width}' style='margin:8px 0; border:1px solid #ddd; padding:4px'/>"


def table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<p style='color:#a00'>[tabela vazia]</p>"
    return df.head(max_rows).to_html(index=False, border=0)


def main(outputs_dir: str = None, report_path: str = None):
    out_dir = Path(outputs_dir or Path(__file__).parent / "outputs")
    report_file = Path(report_path or out_dir / "report.html")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Arquivos esperados
    files_list = out_dir / "files_list.csv"
    feats_csv = out_dir / "image_features.csv"
    feats_meta_csv = out_dir / "image_features_with_meta.csv"
    summary_json = out_dir / "summary.json"
    uni_dir = out_dir / "plots_univariate"
    multi_dir = out_dir / "plots_multivariate"
    fs_dir = out_dir / "feature_selection"

    # Carregamentos tolerantes
    def safe_read_csv(p: Path):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

    files_df = safe_read_csv(files_list)
    feats_df = safe_read_csv(feats_csv)
    meta_df = safe_read_csv(feats_meta_csv)
    try:
        summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json.exists() else {}
    except Exception:
        summary = {}

    # Estatísticas simples
    by_label = meta_df.groupby("label").size().reset_index(name="count") if not meta_df.empty else pd.DataFrame()
    corrupted = int(meta_df.get("is_corrupted", pd.Series(dtype=int)).sum()) if not meta_df.empty else 0

    # Feature selection tables
    mi_csv = fs_dir / "feature_importance_mutual_info.csv"
    rf_csv = fs_dir / "feature_importance_random_forest.csv"
    l1_csv = fs_dir / "feature_importance_logreg_l1.csv"
    mi_df = safe_read_csv(mi_csv)
    rf_df = safe_read_csv(rf_csv)
    l1_df = safe_read_csv(l1_csv)

    # Monta HTML
    html = []
    html.append("""
<!doctype html>
<html lang='pt-br'>
<head>
  <meta charset='utf-8'/>
  <title>EDA Report - TrashNet</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    h1,h2 { color:#222; }
    .card { border:1px solid #eee; padding:16px; border-radius:8px; margin:16px 0; }
    .row { display:flex; gap:24px; flex-wrap: wrap; }
    .col { flex:1 1 420px; }
    table { border-collapse: collapse; }
    th, td { padding:6px 10px; border-bottom: 1px solid #eee; }
    .muted { color:#666; font-size: 13px; }
    code { background:#f6f8fa; padding:2px 4px; border-radius:4px; }
  </style>
</head>
<body>
  <h1>Relatório de EDA — TrashNet</h1>
  <p class='muted'>Este relatório foi gerado a partir das saídas em <code>trashnet/eda/outputs</code>. Imagens e tabelas abaixo referenciam os arquivos gerados pelo script <code>eda.py</code>.</p>
""")

    # Resumo
    html.append("<div class='card'>")
    html.append("<h2>Resumo</h2>")
    total = int(summary.get("num_images", len(meta_df)))
    num_corrupted = int(summary.get("num_corrupted", corrupted))
    with_phash = int(summary.get("num_with_phash", int(meta_df.get("phash", pd.Series(dtype=object)).notna().sum()) if not meta_df.empty else 0))
    html.append(f"<p>Total de imagens: <b>{total}</b> • Corrompidas: <b>{num_corrupted}</b> • Com pHash: <b>{with_phash}</b></p>")
    if not by_label.empty:
        html.append("<h3>Distribuição por classe</h3>")
        html.append(table_html(by_label))
    html.append("</div>")

    # Univariada
    html.append("<div class='card'>")
    html.append("<h2>Análise Univariada</h2>")
    html.append(img_tag(uni_dir / "class_distribution.png", width=600))
    # exibe alguns KDEs se existirem
    for name in [
        "kde_brightness_mean.png","kde_sharpness_lapl_var.png","kde_colorfulness.png","kde_entropy_gray.png",
        "kde_aspect_ratio.png","kde_area_px.png"
    ]:
        p = uni_dir / name
        if p.exists():
            html.append(img_tag(p, width=560))
    html.append("</div>")

    # Multivariada
    html.append("<div class='card'>")
    html.append("<h2>Análise Multivariada</h2>")
    html.append("<div class='row'>")
    html.append("<div class='col'>" + img_tag(multi_dir / "correlation_heatmap.png", width=560) + "</div>")
    html.append("<div class='col'>" + img_tag(multi_dir / "pca_scatter.png", width=560) + "</div>")
    html.append("</div>")
    html.append("</div>")

    # Seleção de atributos
    html.append("<div class='card'>")
    html.append("<h2>Seleção de Atributos</h2>")
    if not mi_df.empty:
        html.append("<h3>Mutual Information</h3>" + table_html(mi_df))
    if not rf_df.empty:
        html.append("<h3>RandomForest Importance</h3>" + table_html(rf_df))
    if not l1_df.empty:
        html.append("<h3>Logistic Regression (L1)</h3>" + table_html(l1_df))
    html.append("</div>")

    html.append("""
  <div class='muted'>
    <p>Gerado por <code>trashnet/eda/make_report.py</code>. Para atualizar, reexecute a EDA e depois rode o gerador de relatório.</p>
  </div>
</body>
</html>
""")

    report_file.write_text("\n".join(html), encoding="utf-8")
    print("[REPORT] Arquivo gerado:", report_file)


if __name__ == "__main__":
    # Uso: python make_report.py [outputs_dir] [report_path]
    outputs = sys.argv[1] if len(sys.argv) > 1 else None
    report = sys.argv[2] if len(sys.argv) > 2 else None
    main(outputs, report)
