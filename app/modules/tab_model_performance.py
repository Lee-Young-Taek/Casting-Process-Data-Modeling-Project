# modules/tab_process_explanation.py
from pathlib import Path

from shiny import ui, reactive, render
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import joblib
import base64

APP_DIR = Path(__file__).resolve().parents[1]
SCORE_V0_PATH = APP_DIR / "data" / "models" / "v0"
SCORE_V1_PATH = APP_DIR / "data" / "models" / "v1"
SCORE_V2_PATH = APP_DIR / "data" / "models" / "v2"

SCORE_V0_LGBM = SCORE_V0_PATH / "LightGBM_v0_scores.csv"
SCORE_V1_LGBM = SCORE_V1_PATH / "LightGBM_v1_scores.csv"
SCORE_V2_LGBM = SCORE_V2_PATH / "LightGBM_v2_scores.csv"

SCORE_V0_RF = SCORE_V0_PATH / "RandomForest_v0_scores.csv"
SCORE_V1_RF = SCORE_V1_PATH / "RandomForest_v1_scores.csv"
SCORE_V2_RF = SCORE_V2_PATH / "RandomForest_v2_scores.csv"

SCORE_V0_XGB = SCORE_V0_PATH / "XGBoost_v0_scores.csv"
SCORE_V1_XGB = SCORE_V1_PATH / "XGBoost_v1_scores.csv"
SCORE_V2_XGB = SCORE_V2_PATH / "XGBoost_v2_scores.csv"

MODELS = ["LightGBM", "RandomForest", "XGBoost"]
VERSIONS = ["v0", "v1", "v2"]

DISPLAY_METRIC_MAP = {
    "ROC-AUC": "roc_auc",
    "F1-Score": "f1_score",
    "Recall": "recall",
    "Precision": "precision",
}

DETAIL_METRIC_FIELDS = [
    ("ROC-AUC", "roc_auc"),
    ("F1-Score", "f1_score"),
    ("Precision", "precision"),
    ("Recall", "recall"),
]

APP_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "data" / "models"

MODEL_METRICS: dict[str, dict[str, dict[str, float]]] = {model: {} for model in MODELS}
METRIC_VALUES: dict[str, dict[str, dict[str, float]]] = {}
HIGHLIGHT_MAP: dict[str, tuple[str, str] | None] = {}
MODEL_ARTIFACT_CACHE: dict[tuple[str, str], dict | None] = {}
FEATURE_IMPORTANCE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
BEST_PARAMS_CACHE: dict[tuple[str, str], dict[str, object]] = {}
PDP_IMAGE_CACHE: dict[tuple[str, str], str | None] = {}
PDP_IMAGE_DIR = APP_DIR / "data" / "png"


def _normalize_artifact_model(artifact: dict | None) -> dict | None:
    if not isinstance(artifact, dict):
        return artifact

    if artifact.get("model") is None:
        model_aliases = [
            "model",
            "model_clf",
            "model_reg",
            "estimator",
            "classifier",
            "regressor",
            "clf",
        ]
        for alias in model_aliases:
            if alias in artifact and artifact[alias] is not None:
                artifact["model"] = artifact[alias]
                break

    return artifact


def _resolve_metric_selection(metric_name: str) -> tuple[str, str] | None:
    highlight = HIGHLIGHT_MAP.get(metric_name)
    if highlight:
        return highlight

    # Fallback: first available model/version combination with data
    for version in VERSIONS:
        for model in MODELS:
            if MODEL_METRICS.get(model, {}).get(version):
                highlight = (model, version)
                HIGHLIGHT_MAP[metric_name] = highlight
                return highlight
    # Absolute fallback: first declared pair
    if MODELS and VERSIONS:
        highlight = (MODELS[0], VERSIONS[0])
        HIGHLIGHT_MAP[metric_name] = highlight
        return highlight

    return None


def _load_model_metrics() -> None:
    """Load model score summaries from data/models directory."""

    global MODEL_METRICS, METRIC_VALUES, HIGHLIGHT_MAP

    MODEL_METRICS = {model: {} for model in MODELS}
    METRIC_VALUES = {metric: {model: {} for model in MODELS} for metric in DISPLAY_METRIC_MAP.keys()}
    HIGHLIGHT_MAP = {}
    _reset_importance_cache()
    BEST_PARAMS_CACHE.clear()
    PDP_IMAGE_CACHE.clear()
    MODEL_ARTIFACT_CACHE.clear()

    for version in VERSIONS:
        version_dir = MODELS_DIR / version
        if not version_dir.exists():
            continue

        for model in MODELS:
            file_name = f"{model}_{version}_scores.csv"
            file_path = version_dir / file_name
            if not file_path.exists():
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue

            if df.empty:
                continue

            row = df.iloc[0].to_dict()
            MODEL_METRICS[model][version] = row

            for display_label, column_name in DISPLAY_METRIC_MAP.items():
                value = row.get(column_name)
                if pd.notna(value):
                    METRIC_VALUES[display_label][model][version] = float(value)

    for display_label, column_name in DISPLAY_METRIC_MAP.items():
        best_value = None
        best_pair: tuple[str, str] | None = None

        for model in MODELS:
            for version in VERSIONS:
                value = MODEL_METRICS.get(model, {}).get(version, {}).get(column_name)
                if value is None or pd.isna(value):
                    continue
                if best_value is None or value > best_value:
                    best_value = value
                    best_pair = (model, version)

        if best_pair:
            HIGHLIGHT_MAP[display_label] = best_pair
        else:
            HIGHLIGHT_MAP[display_label] = None


def _load_model_artifact(model_name: str, version: str) -> dict | None:
    key = (model_name, version)
    if key in MODEL_ARTIFACT_CACHE:
        cached = MODEL_ARTIFACT_CACHE[key]
        return _normalize_artifact_model(cached)

    artifact_path = MODELS_DIR / version / f"{model_name}_{version}.pkl"
    if not artifact_path.exists():
        MODEL_ARTIFACT_CACHE[key] = None
        return None

    try:
        artifact = joblib.load(artifact_path)
    except Exception:
        artifact = None

    if artifact is None:
        MODEL_ARTIFACT_CACHE[key] = None
        return None

    if not isinstance(artifact, dict):
        artifact = {"model": artifact}

    normalized = _normalize_artifact_model(artifact)
    MODEL_ARTIFACT_CACHE[key] = normalized
    return normalized


def _get_feature_names(artifact: dict | None) -> list[str]:
    if not artifact:
        return []

    feature_names: list[str] = []

    scaler = artifact.get("scaler") if isinstance(artifact, dict) else None
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        feature_names.extend(list(scaler.feature_names_in_))

    ohe = artifact.get("onehot_encoder") if isinstance(artifact, dict) else None
    if ohe is not None:
        if hasattr(ohe, "get_feature_names_out"):
            try:
                feature_names.extend(list(ohe.get_feature_names_out()))
            except Exception:
                pass
        if not feature_names:
            input_features = getattr(ohe, "feature_names_in_", None)
            categories = getattr(ohe, "categories_", None)
            if input_features is not None and categories is not None:
                for base_name, cats in zip(input_features, categories):
                    for cat in cats:
                        feature_names.append(f"{base_name}={cat}")

    if not feature_names:
        model = artifact.get("model") if isinstance(artifact, dict) else None
        if hasattr(model, "feature_name"):
            try:
                feature_names = list(model.feature_name())
            except Exception:
                feature_names = []

    return feature_names


def _compute_importances(model, feature_names: list[str]) -> pd.DataFrame:
    if model is None:
        return pd.DataFrame()

    importances: np.ndarray | None = None

    if hasattr(model, "feature_importance"):
        try:
            importances = np.array(model.feature_importance(importance_type="gain"))
        except TypeError:
            importances = np.array(model.feature_importance())
    elif hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if isinstance(coefs, np.ndarray):
            importances = np.mean(np.abs(coefs), axis=0).ravel()

    if importances is None or importances.size == 0:
        return pd.DataFrame()

    if not feature_names or len(feature_names) != importances.size:
        feature_names = [f"feature_{i}" for i in range(importances.size)]

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    df = df.sort_values("importance", ascending=False, ignore_index=True)

    total = df["importance"].sum()
    if total > 0:
        df["normalized"] = df["importance"] / total
    else:
        df["normalized"] = df["importance"]

    return df


def _get_feature_importance(model_name: str, version: str) -> pd.DataFrame:
    cache_key = (model_name, version)
    if cache_key in FEATURE_IMPORTANCE_CACHE:
        return FEATURE_IMPORTANCE_CACHE[cache_key]

    artifact = _load_model_artifact(model_name, version)
    model = None
    if isinstance(artifact, dict):
        model = artifact.get("model")
    elif artifact is not None:
        model = artifact

    feature_names = _get_feature_names(artifact)
    df = _compute_importances(model, feature_names)

    FEATURE_IMPORTANCE_CACHE[cache_key] = df
    return df


def _reset_importance_cache() -> None:
    FEATURE_IMPORTANCE_CACHE.clear()


def _get_best_params(model_name: str, version: str) -> dict[str, object]:
    cache_key = (model_name, version)
    if cache_key in BEST_PARAMS_CACHE:
        return BEST_PARAMS_CACHE[cache_key]

    artifact = _load_model_artifact(model_name, version)
    params: dict[str, object] | None = None

    if isinstance(artifact, dict):
        params = artifact.get("best_params")
        # Some artifacts may store numeric types such as numpy scalars
        if params is not None:
            params = {
                key: value.item() if isinstance(value, np.generic) else value
                for key, value in params.items()
            }

    if params is None:
        model = artifact.get("model") if isinstance(artifact, dict) else artifact
        if model is not None and hasattr(model, "get_params"):
            try:
                params = model.get_params()
            except Exception:
                params = None

    if params is None:
        params = {}

    BEST_PARAMS_CACHE[cache_key] = params
    return params


def _load_pdp_image(model_name: str, version: str) -> str | None:
    cache_key = (model_name, version)
    if cache_key in PDP_IMAGE_CACHE:
        return PDP_IMAGE_CACHE[cache_key]

    candidates = [
        f"{model_name}_{version}_PDP.png",
        f"{model_name}_{version}.png",
        f"{model_name}_PDP.png",
        f"{model_name}.png",
    ]

    fallback_names = {
        "RandomForest": "RF_basic_PDP.png",
    }

    fallback = fallback_names.get(model_name)
    if fallback:
        candidates.append(fallback)

    candidates.append("RF_basic_PDP.png")

    search_dirs = []
    version_dir = PDP_IMAGE_DIR / version if version else None
    if version_dir and version_dir.exists():
        search_dirs.append(version_dir)
    search_dirs.append(PDP_IMAGE_DIR)

    image_data_uri: str | None = None

    for candidate in candidates:
        if not candidate:
            continue

        for directory in search_dirs:
            image_path = directory / candidate
            if not image_path.exists():
                continue

            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
                image_data_uri = f"data:image/png;base64,{encoded}"
                break
            except Exception:
                continue
        if image_data_uri:
            break

    PDP_IMAGE_CACHE[cache_key] = image_data_uri
    return image_data_uri
_load_model_metrics()

custom_css = """
<style>
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border: 1px solid #dee2e6;
    border-radius: 8px;
    overflow: hidden;
}
.metrics-grid > div {
    aspect-ratio: 1 / 1;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid #dee2e6;
    border-bottom: 1px solid #dee2e6;
    font-size: 1.025rem;
    font-weight: 500;
}
.metrics-grid > div:nth-child(4n) {
    border-right: none;
}
.metrics-grid > div:nth-last-child(-n + 4) {
    border-bottom: none;
}
.metrics-grid__header {
    background-color: #f8f9fa;
    font-weight: 600;
}
.metrics-grid__row-header {
    background-color: #fdfdfd;
    font-weight: 600;
}
.metrics-grid__cell--highlight {
    background-color: #2A2D30;
    color: #ffffff;
    font-weight: 700;
}
.metrics-btn.active {
    background-color: #2A2D30 !important;
    color: #ffffff !important;
}
.metrics-btn {
    transition: all 0.2s ease;
}
.metric-detail-card {
    background-color: #ffffff;
    border: 1px solid #e3e6eb;
    border-radius: 16px;
    padding: 20px 24px;
    height: auto;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 12px;
}
.metric-detail-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #212529;
}
.metric-detail-subtitle {
    font-size: 0.95rem;
    font-weight: 500;
    color: #5c636a;
}
.metric-detail-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}
.metric-detail-table th {
    width: 50%;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    color: #495057;
    background-color: #f8f9fa;
    border-top: 1px solid #e3e6eb;
    border-left: 1px solid #e3e6eb;
}
.metric-detail-table td {
    padding: 10px 12px;
    text-align: right;
    font-weight: 600;
    color: #212529;
    border-top: 1px solid #e3e6eb;
    border-right: 1px solid #e3e6eb;
}
.metric-detail-table tr:first-child th,
.metric-detail-table tr:first-child td {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}
.metric-detail-table tr:last-child th {
    border-bottom-left-radius: 12px;
    border-bottom: 1px solid #e3e6eb;
}
.metric-detail-table tr:last-child td {
    border-bottom-right-radius: 12px;
    border-bottom: 1px solid #e3e6eb;
}
.importance-card {
    margin-top: 16px;
    padding: 16px 20px;
    background-color: #ffffff;
    border: 1px solid #e3e6eb;
    border-radius: 16px;
}
.importance-card h4 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: #212529;
}
.insight-tabset .nav-tabs {
    border-bottom: 1px solid #e3e6eb;
    margin-bottom: 0.75rem;
}
.insight-tabset .nav-link {
    font-weight: 600;
    color: #5c636a;
}
.insight-tabset .nav-link.active {
    color: #2A2D30;
    border-color: #e3e6eb #e3e6eb transparent;
}
.insight-tabset {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 520px;
}
.insight-tabset .tab-content {
    flex-grow: 1;
    display: flex;
}
.insight-tabset .tab-content > .tab-pane {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    min-height: 360px;
}
.insight-tabset .tab-content > .tab-pane:not(.active) {
    display: none;
}
.card-no-gap {
    margin-bottom: 0 !important;
}
.panel-equal-row {
    align-items: stretch !important;
}
.card-equal {
    height: 100%;
    min-height: 520px;
    display: flex;
    flex-direction: column;
}
.card-equal .card-body {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
}
</style>
"""


def _build_metrics_grid(metric_name: str) -> str:
    metric_data = METRIC_VALUES.get(metric_name, {})
    if not metric_data:
        metric_data = {}

    highlight_pair = HIGHLIGHT_MAP.get(metric_name)
    if not highlight_pair:
        highlight_pair = _resolve_metric_selection(metric_name)

    cells = ["<div class='metrics-grid mt-3'>"]

    # Header row
    cells.append("<div class='metrics-grid__header'>모델</div>")
    for version in VERSIONS:
        cells.append(f"<div class='metrics-grid__header'>{version}</div>")

    # Body rows
    for model in MODELS:
        cells.append("<div class='metrics-grid__row-header'>{}</div>".format(model))
        for version in VERSIONS:
            value = metric_data.get(model, {}).get(version)
            cell_classes = ["metrics-grid__cell"]

            if highlight_pair and (model, version) == highlight_pair:
                cell_classes.append("metrics-grid__cell--highlight")

            display_value = "-" if value is None else f"{value:.3f}"
            cells.append(
                f"<div class=\"{' '.join(cell_classes)}\">{display_value}</div>"
            )

    cells.append("</div>")
    return "".join(cells)


def _build_metric_button(button_id: str, label: str, active: bool) -> ui.Tag:
    classes = ["flex-fill", "metrics-btn"]
    if active:
        classes.append("active")
    return ui.input_action_button(
        button_id,
        label,
        class_=" ".join(classes),
        style="font-size:0.85rem; white-space:nowrap; width:100%;",
    )


def _build_selection_details(metric_name: str) -> ui.Tag:
    highlight = _resolve_metric_selection(metric_name)
    if not highlight:
        return ui.div("선택된 모델 정보가 없습니다.", class_="text-muted")

    model_name, version = highlight
    metric_rows = []
    metrics_dict = MODEL_METRICS.get(model_name, {}).get(version, {})

    for display_label, column_name in DETAIL_METRIC_FIELDS:
        value = metrics_dict.get(column_name)
        if value is None or pd.isna(value):
            display_value = "-"
        else:
            display_value = f"{float(value):.3f}"
        metric_rows.append(
            ui.tags.tr(
                ui.tags.th(display_label),
                ui.tags.td(display_value),
            )
        )

    return ui.div(
        ui.div(f"{model_name}", class_="metric-detail-header"),
        ui.div(f"버전: {version}", class_="metric-detail-subtitle"),
        ui.tags.table(
            {"class": "metric-detail-table"},
            *metric_rows,
        ),
        class_="metric-detail-card",
    )

def _format_feature_label(label: str) -> str:
    return label.replace("__", " → ") if "__" in label else label


def _format_param_name(name: str) -> str:
    return name.replace("_", " ").title()


def _format_param_value(value: object) -> str:
    if isinstance(value, (np.floating, float)):
        float_val = float(value)
        if float_val.is_integer():
            return str(int(float_val))
        return f"{float_val:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _build_importance_tab(model_name: str, version: str) -> ui.Tag:
    importance_df = _get_feature_importance(model_name, version)

    if importance_df.empty:
        return ui.div("변수 중요도 정보를 불러올 수 없습니다.", class_="text-muted importance-card")

    top_df = importance_df.head(5).copy()
    top_df["feature"] = top_df["feature"].map(_format_feature_label)
    plot_df = top_df.sort_values("normalized", ascending=True, ignore_index=True)

    def _format_percent(value: float) -> str:
        return f"{value * 100:.1f}%"

    fig = px.bar(
        plot_df,
        x="normalized",
        y="feature",
        orientation="h",
        title=None,
        labels={"feature": "변수", "normalized": "중요도 비율"},
        text=plot_df["normalized"].map(_format_percent),
    )
    max_range = float(min(1.0, max(plot_df["normalized"].max() * 1.15, 0.2)))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange=True),
        xaxis=dict(range=[0, max_range]),
        plot_bgcolor="white",
        showlegend=False,
    )
    fig.update_traces(marker_color="#2A2D30", textposition="outside", cliponaxis=False)

    return ui.div(
        ui.HTML(
            pio.to_html(
                fig,
                include_plotlyjs=True,
                full_html=False,
                config={"displayModeBar": False},
            )
        ),
        class_="importance-card",
    )


def _build_best_params_tab(model_name: str, version: str) -> ui.Tag:
    params = _get_best_params(model_name, version)

    if not params:
        return ui.div("베스트 파라미터 정보를 불러올 수 없습니다.", class_="text-muted importance-card")

    rows = []

    for key in sorted(params.keys()):
        value = params[key]
        rows.append(
            ui.tags.tr(
                ui.tags.th(_format_param_name(key)),
                ui.tags.td(_format_param_value(value)),
            )
        )

    return ui.div(
        ui.tags.table(
            {"class": "metric-detail-table"},
            *rows,
        ),
        class_="importance-card",
    )


def _build_pdp_tab(model_name: str, version: str) -> ui.Tag:
    image_data_uri = _load_pdp_image(model_name, version)

    if not image_data_uri:
        return ui.div("PDP 이미지를 불러올 수 없습니다.", class_="text-muted importance-card")

    alt_text = f"{model_name} {version} 모델의 부분 의존성(PDP) 시각화"

    return ui.div(
        ui.tags.img(
            src=image_data_uri,
            alt=alt_text,
            style="width: 100%; height: auto; border-radius: 12px; border: 1px solid #e3e6eb;",
        ),
        class_="importance-card",
    )


def _build_insight_tabs(model_name: str, version: str) -> ui.Tag:
    navset = ui.navset_tab(
        ui.nav_panel("변수 중요도", _build_importance_tab(model_name, version), value="importance"),
        ui.nav_panel("베스트 파라미터", _build_best_params_tab(model_name, version), value="best_params"),
        ui.nav_panel("PDP", _build_pdp_tab(model_name, version), value="pdp"),
        id="insight_nav",
        selected="importance",
    )
    return ui.div(navset, class_="insight-tabset")

def panel_body():
    return ui.TagList(
        ui.HTML(custom_css),
        ui.div(
            ui.div(
                ui.card(
                    ui.card_header("모델 선택"),
                    ui.card_body(
                        ui.output_ui("metric_button_row"),
                        ui.output_ui("metrics_grid"),
                        class_="w-100 d-flex flex-column gap-3 flex-grow-1",
                    ),
                    class_="card-no-gap card-equal",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            ui.div(
                ui.card(
                    ui.card_header("모델 정보"),
                    ui.card_body(
                        ui.output_ui("selection_details"),
                        ui.output_ui("insight_tabs"),
                        class_="w-100 d-flex flex-column gap-3 flex-grow-1",
                    ),
                    class_="card-no-gap card-equal",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            class_="d-flex flex-row align-items-stretch panel-equal-row w-100",
            style="gap: 1rem; width: 100%;",
        ),
    )


def panel():
    return ui.nav_panel("모델 성능 평가", panel_body())
def server(input, output, session):
    _load_model_metrics()
    active_metric = reactive.Value("F1-Score")

    @reactive.effect
    @reactive.event(input.btn_metric_roc_auc)
    def _set_metric_roc_auc():
        active_metric.set("ROC-AUC")

    @reactive.effect
    @reactive.event(input.btn_metric_f1)
    def _set_metric_f1():
        active_metric.set("F1-Score")

    @reactive.effect
    @reactive.event(input.btn_metric_recall)
    def _set_metric_recall():
        active_metric.set("Recall")

    @reactive.effect
    @reactive.event(input.btn_metric_precision)
    def _set_metric_precision():
        active_metric.set("Precision")

    @render.ui
    def metrics_grid():
        metric_name = active_metric.get()
        return ui.HTML(_build_metrics_grid(metric_name))

    @render.ui
    def metric_button_row():
        metric_name = active_metric.get()
        return ui.div(
            _build_metric_button("btn_metric_roc_auc", "ROC-AUC", metric_name == "ROC-AUC"),
            _build_metric_button("btn_metric_f1", "F1-Score", metric_name == "F1-Score"),
            _build_metric_button("btn_metric_recall", "Recall", metric_name == "Recall"),
            _build_metric_button("btn_metric_precision", "Precision", metric_name == "Precision"),
            class_="d-flex w-100 gap-2",
        )

    @render.ui
    def selection_details():
        metric_name = active_metric.get()
        return _build_selection_details(metric_name)

    @render.ui
    def insight_tabs():
        metric_name = active_metric.get()
        highlight = _resolve_metric_selection(metric_name)
        if not highlight:
            return ui.div("세부 인사이트 정보가 없습니다.", class_="text-muted")

        model_name, version = highlight
        return _build_insight_tabs(model_name, version)

