from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# --- 경로 및 자원 로드 ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "processed" / "test_v1.csv"
MODEL_FILE = BASE_DIR / "data" / "models" / "v1" / "LightGBM_v1.pkl"

df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)
artifact = joblib.load(MODEL_FILE)

lgb_model = artifact["model"]
scaler = artifact.get("scaler")
ordinal_encoder = artifact.get("ordinal_encoder")
onehot_encoder = artifact.get("onehot_encoder")
operating_threshold = float(artifact.get("operating_threshold", 0.5))

numeric_cols_model = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else []
categorical_cols_model = list(getattr(ordinal_encoder, "feature_names_in_", [])) if ordinal_encoder is not None else []
all_model_cols = numeric_cols_model + categorical_cols_model

missing_columns = [col for col in all_model_cols if col not in df.columns]
for col in missing_columns:
    df[col] = np.nan

numeric_cols = list(numeric_cols_model)
categorical_cols = list(categorical_cols_model)
all_cols = list(all_model_cols)

FORCED_CATEGORICAL_COLS = {'mold_code', 'EMS_operation_time'}

categorical_cols_ui = list(categorical_cols)
for col in FORCED_CATEGORICAL_COLS:
    if col in all_cols and col not in categorical_cols_ui:
        categorical_cols_ui.append(col)

numeric_cols_ui = [col for col in numeric_cols if col not in FORCED_CATEGORICAL_COLS]

COLUMN_NAMES_KR = {
    "registration_time": "등록 일시",
    "count": "생산 순번",
    "working": "가동 여부",
    "emergency_stop": "비상 정지",
    "facility_operation_cycleTime": "설비 운영 사이클타임",
    "production_cycletime": "제품 생산 사이클타임",
    "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도",
    "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스킷 두께",
    "upper_mold_temp1": "상부 금형 온도1",
    "upper_mold_temp2": "상부 금형 온도2",
    "lower_mold_temp1": "하부 금형 온도1",
    "lower_mold_temp2": "하부 금형 온도2",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "전자교반 가동시간",
    "mold_code": "금형 코드",
    "tryshot_signal": "트라이샷 신호",
    "molten_temp": "용탕 온도",
    "uniformity": "균일도",
    "mold_temp_udiff": "금형 온도차(상/하)",
    "P_diff": "압력 차이",
    "Cycle_diff": "사이클 시간 차이",
    "hour":"시간",
    "weekday":"일자"
}
pass_reference = df[df.get("passorfail", 0) == 0].copy()
if pass_reference.empty:
    pass_reference = df.copy()


NUMERIC_FEATURE_RANGES = {}
for col in numeric_cols_ui:
    series = pass_reference[col].dropna()
    if series.empty:
        series = df[col].dropna()
    if not series.empty:
        NUMERIC_FEATURE_RANGES[col] = (float(series.min()), float(series.max()))
MIN_RECOMMENDATION_SPAN_RATIO = 0.15
MIN_RECOMMENDATION_SPAN_ABS = 1.0


explainer = shap.TreeExplainer(lgb_model)

def _resolve_expected_value(expected):
    if isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected) > 1:
            return float(expected[1])
        return float(expected[0])
    return float(expected)

SHAP_EXPECTED_VALUE = _resolve_expected_value(explainer.expected_value)

numeric_index_map = {feat: idx for idx, feat in enumerate(numeric_cols)}

ohe_feature_slices = {}
ohe_value_labels = {}
start_idx = len(numeric_cols)
if categorical_cols and onehot_encoder is not None and ordinal_encoder is not None:
    for feat, ohe_cats, ord_cats in zip(categorical_cols, onehot_encoder.categories_, ordinal_encoder.categories_):
        length = len(ohe_cats)
        ohe_feature_slices[feat] = (start_idx, start_idx + length)
        labels = []
        for code in ohe_cats:
            code_int = int(code)
            if 0 <= code_int < len(ord_cats):
                labels.append(str(ord_cats[code_int]))
            else:
                labels.append("unknown")
        ohe_value_labels[feat] = labels
        start_idx += length
else:
    for feat in categorical_cols:
        ohe_feature_slices[feat] = (len(numeric_cols), len(numeric_cols))
        ohe_value_labels[feat] = []

def _aggregate_shap_vector(shap_vector):
    contributions = {}
    for feat, idx in numeric_index_map.items():
        contributions[feat] = float(shap_vector[idx])
    for feat, (start, end) in ohe_feature_slices.items():
        if end > start:
            contributions[feat] = float(np.sum(shap_vector[start:end]))
        else:
            contributions[feat] = 0.0
    return contributions


def compute_shap_summary_data(features, reference_df=None, max_samples=400):
    if not features:
        return None
    source_df = reference_df
    if source_df is None or source_df.empty:
        source_df = pass_reference if not pass_reference.empty else df
    if source_df.empty:
        return None
    if len(source_df) > max_samples:
        sample_df = source_df.sample(n=max_samples, random_state=42).copy()
    else:
        sample_df = source_df.copy()
    if sample_df.empty:
        return None
    working_df = sample_df[all_cols].copy() if all_cols else sample_df.copy()
    if numeric_cols:
        working_df[numeric_cols] = working_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        working_df[categorical_cols] = working_df[categorical_cols].fillna("unknown").astype(str)
    feature_matrix = prepare_feature_matrix(working_df)
    shap_values_raw = explainer.shap_values(feature_matrix)
    if isinstance(shap_values_raw, list):
        shap_matrix = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
    else:
        shap_matrix = shap_values_raw
    if shap_matrix is None or len(shap_matrix) == 0:
        return None
    shap_matrix = np.asarray(shap_matrix)
    aggregated = {feat: [] for feat in features}
    for shap_vector in shap_matrix:
        agg = _aggregate_shap_vector(shap_vector)
        for feat in features:
            aggregated[feat].append(float(agg.get(feat, 0.0)))
    values_map = {}
    for feat in features:
        if feat in sample_df.columns:
            values_map[feat] = sample_df[feat].tolist()
        else:
            values_map[feat] = [np.nan] * len(sample_df)
    return {"shap": aggregated, "values": values_map, "count": len(sample_df)}

def compute_feature_distribution_data(features, max_samples=400):
    if not features:
        return None
    if "passorfail" in df.columns:
        pass_df = pass_reference if not pass_reference.empty else df[df.get("passorfail") == 0]
        fail_df = df[df.get("passorfail") == 1]
    else:
        pass_df = pass_reference if not pass_reference.empty else df
        fail_df = df[df.index.isin([])]
    if pass_df is None or pass_df.empty:
        pass_df = df
    def _take_sample(source):
        if source is None or source.empty:
            return source
        if len(source) > max_samples:
            return source.sample(n=max_samples, random_state=42)
        return source
    pass_sample = _take_sample(pass_df)
    fail_sample = _take_sample(fail_df)
    result = {}
    for feat in features:
        if feat not in df.columns:
            continue
        feature_type = "numeric" if feat in numeric_cols else ("categorical" if feat in categorical_cols else "unknown")
        entry = {"type": feature_type}
        if feature_type == "numeric":
            pass_vals = pd.to_numeric(pass_sample.get(feat), errors="coerce") if pass_sample is not None else pd.Series(dtype=float)
            fail_vals = pd.to_numeric(fail_sample.get(feat), errors="coerce") if fail_sample is not None else pd.Series(dtype=float)
            entry["pass"] = pass_vals.dropna().tolist() if pass_vals is not None else []
            entry["fail"] = fail_vals.dropna().tolist() if fail_vals is not None else []
        else:
            pass_vals = pass_sample.get(feat) if pass_sample is not None else pd.Series(dtype=object)
            fail_vals = fail_sample.get(feat) if fail_sample is not None else pd.Series(dtype=object)
            pass_list = pass_vals.fillna("missing").astype(str).tolist() if pass_vals is not None else []
            fail_list = fail_vals.fillna("missing").astype(str).tolist() if fail_vals is not None else []
            entry["pass"] = pass_list
            entry["fail"] = fail_list
        result[feat] = entry
    return result

    if not features:
        return None
    source_df = reference_df
    if source_df is None or source_df.empty:
        source_df = pass_reference if not pass_reference.empty else df
    if source_df.empty:
        return None
    if len(source_df) > max_samples:
        sample_df = source_df.sample(n=max_samples, random_state=42).copy()
    else:
        sample_df = source_df.copy()
    if sample_df.empty:
        return None
    working_df = sample_df[all_cols].copy() if all_cols else sample_df.copy()
    if numeric_cols:
        working_df[numeric_cols] = working_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        working_df[categorical_cols] = working_df[categorical_cols].fillna("unknown").astype(str)
    feature_matrix = prepare_feature_matrix(working_df)
    shap_values_raw = explainer.shap_values(feature_matrix)
    if isinstance(shap_values_raw, list):
        shap_matrix = shap_values_raw[1] if len(shap_values_raw) > 1 else shap_values_raw[0]
    else:
        shap_matrix = shap_values_raw
    if shap_matrix is None or len(shap_matrix) == 0:
        return None
    shap_matrix = np.asarray(shap_matrix)
    aggregated = {feat: [] for feat in features}
    for shap_vector in shap_matrix:
        agg = _aggregate_shap_vector(shap_vector)
        for feat in features:
            aggregated[feat].append(float(agg.get(feat, 0.0)))
    values_map = {}
    for feat in features:
        if feat in sample_df.columns:
            values_map[feat] = sample_df[feat].tolist()
        else:
            values_map[feat] = [np.nan] * len(sample_df)
    return {"shap": aggregated, "values": values_map, "count": len(sample_df)}


def build_input_dataframe(row_dict):
    data = {col: row_dict.get(col) for col in all_cols}
    input_df = pd.DataFrame([data], columns=all_cols)
    if numeric_cols:
        input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        input_df[categorical_cols] = input_df[categorical_cols].fillna("unknown").astype(str)
    return input_df

def prepare_feature_matrix(input_df):
    arrays = []
    if numeric_cols:
        if scaler is not None:
            num_part = scaler.transform(input_df[numeric_cols].astype(float))
        else:
            num_part = input_df[numeric_cols].to_numpy(dtype=float)
        arrays.append(num_part)
    if categorical_cols:
        if ordinal_encoder is not None and onehot_encoder is not None:
            cat_values = input_df[categorical_cols]
            cat_ord = ordinal_encoder.transform(cat_values).astype(int)
            cat_ohe = onehot_encoder.transform(cat_ord)
            if hasattr(cat_ohe, "toarray"):
                cat_ohe = cat_ohe.toarray()
        else:
            cat_ohe = np.zeros((len(input_df), 0))
        arrays.append(cat_ohe)
    if arrays:
        return np.hstack(arrays).astype(np.float32)
    return np.zeros((len(input_df), 0), dtype=np.float32)

def compute_shap_contributions(feature_matrix):
    shap_values = explainer.shap_values(feature_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_vector = shap_values[0]
    contributions = _aggregate_shap_vector(shap_vector)
    return contributions, shap_vector

def predict_with_model(row_dict, compute_shap=False):
    input_df = build_input_dataframe(row_dict)
    feature_matrix = prepare_feature_matrix(input_df)
    if hasattr(lgb_model, "predict_proba"):
        proba = lgb_model.predict_proba(feature_matrix)
        probability = float(proba[0, 1] if proba.ndim == 2 else proba[0])
    else:
        raw_pred = lgb_model.predict(feature_matrix)
        probability = float(raw_pred[0] if raw_pred.ndim else raw_pred)
    prediction = 1 if probability >= operating_threshold else 0
    forced_fail = False
    tryshot = row_dict.get("tryshot_signal")
    if tryshot is not None and str(tryshot).upper() == "D":
        if probability < operating_threshold:
            forced_fail = True
        prediction = 1
    result = {
        "probability": probability,
        "prediction": prediction,
        "forced_fail": forced_fail,
        "input_df": input_df,
        "features": feature_matrix
    }
    if compute_shap:
        contributions, shap_vector = compute_shap_contributions(feature_matrix)
        result["shap_aggregated"] = contributions
        result["shap_vector"] = shap_vector
    return result

def _extract_feature_values(row_dict):
    values = []
    for col in all_cols:
        val = row_dict.get(col, np.nan)
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else np.nan
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            val = str(val)
        if pd.isna(val):
            values.append(np.nan)
        else:
            values.append(val)
    return values


def build_shap_explanation(contributions, input_row):
    if not contributions:
        return None
    shap_values = np.array([float(contributions.get(col, 0.0)) for col in all_cols], dtype=float)
    feature_values = np.array(_extract_feature_values(input_row), dtype=object)
    feature_names = [COLUMN_NAMES_KR.get(col, col) for col in all_cols]
    try:
        return shap.Explanation(values=shap_values, base_values=SHAP_EXPECTED_VALUE, data=feature_values, feature_names=feature_names)
    except Exception:
        return None


def evaluate_prediction(row_dict):
    result = predict_with_model(row_dict, compute_shap=False)
    return result["prediction"], result["probability"]


def find_normal_range_binary_fixed(base_row, feature, bounds, threshold=operating_threshold, tol_ratio=0.01, max_iter=20, n_check=5, min_span_ratio=0.1, min_span_absolute=1.0):
    if not bounds:
        return None
    f_min, f_max = bounds
    if pd.isna(f_min) or pd.isna(f_max):
        return None
    low, high = float(f_min), float(f_max)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return None
    orig_low, orig_high = low, high
    orig_span = orig_high - orig_low
    tol = max((high - low) * tol_ratio, 1e-3)
    best_details = None
    for _ in range(max_iter):
        samples = np.linspace(low, high, n_check)
        normal_samples = []
        for val in samples:
            trial = base_row.copy()
            trial[feature] = float(val)
            pred, prob = evaluate_prediction(trial)
            if pred == 0:
                normal_samples.append((float(val), float(prob)))
        if not normal_samples:
            break
        normal_samples.sort(key=lambda item: item[1])
        low = min(v for v, _ in normal_samples)
        high = max(v for v, _ in normal_samples)
        top_val, top_prob = normal_samples[0]
        examples = [top_val]
        if low not in examples:
            examples.append(low)
        if high not in examples:
            examples.append(high)
        if best_details is None or top_prob < best_details[3]:
            best_details = (low, high, examples[:3], top_prob)
        if (high - low) <= tol:
            break
    if best_details is None:
        return None
    low, high, examples, best_prob = best_details
    min_span = max(orig_span * min_span_ratio, min_span_absolute)
    if (high - low) < min_span:
        center = (high + low) / 2.0
        low = max(orig_low, center - min_span / 2.0)
        high = min(orig_high, center + min_span / 2.0)
    examples = list({float(v) for v in (examples + [low, high])})
    examples.sort()
    return {
        "min": float(low),
        "max": float(high),
        "examples": examples[:3],
        "best_prob": float(best_prob)
    }


def binary_search_normal_ranges(base_row, features, feature_ranges, threshold=operating_threshold, max_iter=10, tol_ratio=0.01, min_span_ratio=0.1, min_span_absolute=1.0):
    usable = {}
    base_ranges = {}
    for feat in features:
        bounds = feature_ranges.get(feat)
        if not bounds:
            continue
        f_min, f_max = bounds
        if pd.isna(f_min) or pd.isna(f_max) or not np.isfinite(f_min) or not np.isfinite(f_max):
            continue
        if f_min >= f_max:
            continue
        low_val, high_val = float(f_min), float(f_max)
        usable[feat] = [low_val, high_val]
        base_ranges[feat] = (low_val, high_val)
    if not usable:
        return None, {}, None
    best_solution = None
    best_prob = None
    for _ in range(max_iter):
        trial = base_row.copy()
        mids = {}
        for feat, (low, high) in usable.items():
            mid = (low + high) / 2.0
            mids[feat] = mid
            trial[feat] = mid
        pred, prob = evaluate_prediction(trial)
        is_normal = pred == 0
        if is_normal and (best_prob is None or prob < best_prob):
            best_prob = float(prob)
            best_solution = {feat: float(val) for feat, val in mids.items()}
        updated = False
        for feat, (low, high) in list(usable.items()):
            mid = mids[feat]
            left = mid - low
            right = high - mid
            if left <= 0 and right <= 0:
                continue
            base_range = feature_ranges[feat]
            tol = max((base_range[1] - base_range[0]) * tol_ratio, 1e-3)
            if is_normal:
                new_range = [low, mid] if left >= right else [mid, high]
            else:
                new_range = [mid, high] if left >= right else [low, mid]
            if abs(new_range[1] - new_range[0]) < tol:
                new_range = [float(new_range[0]), float(new_range[1])]
            if new_range != usable[feat]:
                usable[feat] = new_range
                updated = True
        if not updated:
            break
    adjusted_ranges = {}
    for feat, bounds in usable.items():
        base_low, base_high = base_ranges.get(feat, (bounds[0], bounds[1]))
        base_span = base_high - base_low
        cur_span = bounds[1] - bounds[0]
        if base_span > 0:
            min_span = max(base_span * min_span_ratio, min_span_absolute)
            if cur_span < min_span:
                center = (bounds[0] + bounds[1]) / 2.0
                new_low = max(base_low, center - min_span / 2.0)
                new_high = min(base_high, center + min_span / 2.0)
                bounds = [new_low, new_high]
        adjusted_ranges[feat] = (float(bounds[0]), float(bounds[1]))
    return best_solution, adjusted_ranges, best_prob


def evaluate_categorical_candidates(base_row, feature, choices, top_k=3):
    candidates = []
    for value in choices:
        trial = base_row.copy()
        trial[feature] = value
        pred, prob = evaluate_prediction(trial)
        if pred == 0:
            candidates.append((value, float(prob)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1])
    values = [val for val, _ in candidates[:top_k]]
    return {
        "values": values,
        "best_prob": float(candidates[0][1])
    }


def recommend_ranges(base_row, focus_features):
    if not focus_features:
        return {}
    recommendations = {}
    best_prob = None

    numeric_targets = [feat for feat in focus_features if feat in numeric_cols_ui]
    categorical_targets = [feat for feat in focus_features if feat in categorical_cols_ui]

    numeric_ranges = {feat: NUMERIC_FEATURE_RANGES.get(feat) for feat in numeric_targets if NUMERIC_FEATURE_RANGES.get(feat)}

    if len(numeric_ranges) >= 2:
        solution, final_ranges, prob_multi = binary_search_normal_ranges(base_row, list(numeric_ranges.keys()), numeric_ranges, threshold=operating_threshold, min_span_ratio=MIN_RECOMMENDATION_SPAN_RATIO, min_span_absolute=MIN_RECOMMENDATION_SPAN_ABS)
        if solution:
            for feat, mid in solution.items():
                bounds = final_ranges.get(feat, numeric_ranges.get(feat))
                if not bounds:
                    continue
                record = recommendations.get(feat, {"type": "numeric"})
                record["min"] = float(bounds[0])
                record["max"] = float(bounds[1])
                examples = record.get("examples", [])
                if mid not in examples:
                    examples.append(float(mid))
                record["examples"] = examples[:3]
                record["method"] = "binary_multi"
                recommendations[feat] = record
            if prob_multi is not None:
                best_prob = prob_multi if best_prob is None else min(best_prob, prob_multi)

    for feat, bounds in numeric_ranges.items():
        details = find_normal_range_binary_fixed(base_row, feat, bounds, threshold=operating_threshold, min_span_ratio=MIN_RECOMMENDATION_SPAN_RATIO, min_span_absolute=MIN_RECOMMENDATION_SPAN_ABS)
        if not details:
            continue
        record = recommendations.get(feat, {"type": "numeric"})
        record["min"] = details["min"]
        record["max"] = details["max"]
        examples = record.get("examples", [])
        for val in details.get("examples", []):
            if val not in examples:
                examples.append(val)
        record["examples"] = examples[:3]
        record["method"] = record.get("method", "binary_search")
        recommendations[feat] = record
        prob_val = details.get("best_prob")
        if prob_val is not None:
            best_prob = prob_val if best_prob is None else min(best_prob, prob_val)

    for feat in categorical_targets:
        meta = input_metadata.get(feat)
        if not meta:
            continue
        choices = meta.get("choices", [])
        result = evaluate_categorical_candidates(base_row, feat, choices, top_k=3)
        if not result:
            continue
        recommendations[feat] = {
            "type": "categorical",
            "values": result["values"]
        }
        best_prob = result["best_prob"] if best_prob is None else min(best_prob, result["best_prob"])

    if best_prob is not None:
        recommendations["best_probability"] = float(best_prob)
    return recommendations



def format_value(value):
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)

custom_css = """
<style>
body {
    font-family: -apple-system, sans-serif;
    background-color: #f5f7fa;
}
.accordion-section {
    background: white;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    overflow: hidden;
}
.accordion-header {
    background: #2A2D30;
    color: white;
    padding: 20px 28px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    border: none;
    width: 100%;
    text-align: left;
    font-size: 16px;
    font-weight: 600;
    border-radius: 16px 16px 0 0;
}
.accordion-header:hover { background-color: #3B3E42; }
.accordion-content {
    padding: 24px 28px;
    background: #ffffff;
    border-radius: 0 0 16px 16px;
}
.input-item {
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.normal-range-bar {
    display: none;
}
.normal-range-fill {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 0%;
    background: #0d6efd;
    border-radius: 4px;
    transition: left 0.2s ease, width 0.2s ease;
}
.normal-range-label {
    display: none;
}
.irs--shiny .irs-bar { background: #2A2D30; }
.irs--shiny .irs-handle { border: 2px solid #142D4A; background: white; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single { background: #2A2D30; }
#predict:hover { background: #b91f1f !important; transform: translateY(-1px); }
#load_defect_sample:hover { background: #a8a6a6 !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(194, 192, 192, 0.4) !important; }
#apply_recommendations:hover { background: #7cd96a !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(124, 217, 106, 0.4) !important; }
.hidden { display: none !IMPORTANT; }
#settings-button svg {
    transition: transform 0.2s;
}
#settings-button.open svg {
    transform: rotate(180deg);
}
#draggable-prediction {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
#draggable-prediction .card-header {
    border-radius: 16px 16px 0 0 !important;
}
.shap-plot-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px;
    margin-bottom: 16px;
}
.shap-plot-card h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
    color: #142D4A;
}
.shap-plot-empty {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 24px 12px;
}

.shap-plot-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
</style>
"""
custom_js = """
<script>
(function() {
    const FAIL_COLOR = '#dc3545';
    const NORMAL_COLOR = '#0d6efd';
    const BASE_COLOR = '#f8d7da';
    const HANDLER_ID = 'update-normal-range';
    const MAX_ATTEMPTS = 120;
    const RETRY_DELAY = 100;
    const APPLY_RETRY_LIMIT = 25;
    const APPLY_RETRY_DELAY = 120;

    function resetBar(id) {
        const bar = document.getElementById(id + '_normal_bar');
        const primary = document.getElementById(id + '_normal_fill');
        const label = document.getElementById(id + '_normal_label');
        if (bar) {
            bar.style.background = BASE_COLOR;
        }
        if (primary) {
            primary.style.left = '0%';
            primary.style.width = '0%';
        }
        if (label) {
            label.textContent = '';
        }
    }

    function applyBarRange(id, info, attempt = 0) {
        const bar = document.getElementById(id + '_normal_bar');
        const primary = document.getElementById(id + '_normal_fill');
        const label = document.getElementById(id + '_normal_label');
        if (!bar || !primary) {
            if (attempt < APPLY_RETRY_LIMIT) {
                setTimeout(() => applyBarRange(id, info, attempt + 1), APPLY_RETRY_DELAY);
            } else {
                console.warn('normal-range:element-missing', { id, info });
            }
            return;
        }
        const start = Number(info.start_pct);
        const width = Number(info.width_pct);
        if (!Number.isFinite(start) || !Number.isFinite(width)) {
            resetBar(id);
            return;
        }
        const safeStart = Math.max(0, Math.min(100, start));
        let safeWidth = Math.max(0, Math.min(100, width));
        if (safeStart + safeWidth > 100) {
            safeWidth = 100 - safeStart;
        }
        const end = safeStart + safeWidth;
        if (primary) {
            primary.style.left = safeStart + '%';
            primary.style.width = safeWidth + '%';
        }
        if (bar) {
            bar.style.background = `linear-gradient(to right, ${FAIL_COLOR} 0%, ${FAIL_COLOR} ${safeStart}%, ${NORMAL_COLOR} ${safeStart}%, ${NORMAL_COLOR} ${end}%, ${FAIL_COLOR} ${end}%, ${FAIL_COLOR} 100%)`;
        }
        if (label && typeof info.label_text === 'string') {
            label.textContent = info.label_text;
        }
        console.debug('normal-range:update', { id, start: safeStart, width: safeWidth, info, attempt });
    }

    function registerHandler() {
        if (!window.Shiny || !Shiny.addCustomMessageHandler) {
            return false;
        }
        if (window.__normalRangeHandlerRegistered) {
            return true;
        }
        Shiny.addCustomMessageHandler(HANDLER_ID, function(message) {
            const features = message.features || [];
            console.debug('normal-range:reset', features);
            features.forEach(function(id) {
                resetBar(id);
            });
            const ranges = message.ranges || {};
            Object.keys(ranges).forEach(function(id) {
                applyBarRange(id, ranges[id]);
            });
        });
        window.__normalRangeHandlerRegistered = true;
        console.debug('normal-range:handler-registered');
        return true;
    }

    if (!registerHandler()) {
        let attempts = 0;
        const timer = setInterval(function() {
            attempts += 1;
            if (registerHandler() || attempts >= MAX_ATTEMPTS) {
                clearInterval(timer);
                if (attempts >= MAX_ATTEMPTS) {
                    console.warn('normal-range:failed-to-register');
                }
            }
        }, RETRY_DELAY);
        document.addEventListener('shiny:connected', function onConnect() {
            if (registerHandler()) {
                clearInterval(timer);
            }
        }, { once: true });
    }

    function bindPredictionToggle(attempt = 0) {
        const button = document.getElementById('settings-button');
        const panel = document.getElementById('draggable-prediction');
        if (button && panel) {
            if (button.dataset.toggleBound === '1') {
                return true;
            }
            button.dataset.toggleBound = '1';
            button.classList.toggle('open', !panel.classList.contains('hidden'));
            button.addEventListener('click', function() {
                const isHidden = panel.classList.toggle('hidden');
                button.classList.toggle('open', !isHidden);
            });
            return true;
        }
        if (attempt >= MAX_ATTEMPTS) {
            console.warn('prediction-toggle:failed-to-bind');
            return false;
        }
        setTimeout(function() {
            bindPredictionToggle(attempt + 1);
        }, RETRY_DELAY);
        return false;
    }

    function setupPredictionDrag(attempt = 0) {
        const panel = document.getElementById('draggable-prediction');
        const header = panel ? panel.querySelector('.card-header') : null;
        if (panel && header) {
            if (panel.dataset.dragBound === '1') {
                return true;
            }
            panel.dataset.dragBound = '1';
            let dragging = false;
            let offsetX = 0;
            let offsetY = 0;

            function stopDrag() {
                if (!dragging) {
                    return;
                }
                dragging = false;
                document.removeEventListener('mousemove', onPointerMove);
                document.removeEventListener('mouseup', stopDrag);
                document.removeEventListener('touchmove', onPointerMove);
                document.removeEventListener('touchend', stopDrag);
                document.removeEventListener('touchcancel', stopDrag);
            }

            function onPointerMove(event) {
                if (!dragging) {
                    return;
                }
                const point = event.touches && event.touches[0] ? event.touches[0] : event;
                const clientX = point.clientX;
                const clientY = point.clientY;
                if (typeof clientX !== 'number' || typeof clientY !== 'number') {
                    return;
                }
                const maxLeft = Math.max(0, window.innerWidth - panel.offsetWidth);
                const maxTop = Math.max(0, window.innerHeight - panel.offsetHeight);
                const nextLeft = Math.min(Math.max(0, clientX - offsetX), maxLeft);
                const nextTop = Math.min(Math.max(0, clientY - offsetY), maxTop);
                panel.style.left = nextLeft + 'px';
                panel.style.top = nextTop + 'px';
                if (event.cancelable) {
                    event.preventDefault();
                }
            }

            function startDrag(event) {
                const point = event.touches && event.touches[0] ? event.touches[0] : event;
                const rect = panel.getBoundingClientRect();
                offsetX = point.clientX - rect.left;
                offsetY = point.clientY - rect.top;
                panel.style.right = 'auto';
                panel.style.bottom = 'auto';
                panel.style.left = rect.left + 'px';
                panel.style.top = rect.top + 'px';
                dragging = true;
                document.addEventListener('mousemove', onPointerMove);
                document.addEventListener('mouseup', stopDrag);
                document.addEventListener('touchmove', onPointerMove, { passive: false });
                document.addEventListener('touchend', stopDrag);
                document.addEventListener('touchcancel', stopDrag);
                if (event.cancelable) {
                    event.preventDefault();
                }
            }

            header.addEventListener('mousedown', startDrag);
            header.addEventListener('touchstart', startDrag, { passive: false });
            return true;
        }
        if (attempt >= MAX_ATTEMPTS) {
            console.warn('prediction-drag:failed-to-bind');
            return false;
        }
        setTimeout(function() {
            setupPredictionDrag(attempt + 1);
        }, RETRY_DELAY);
        return false;
    }

    function initializePredictionPanel() {
        bindPredictionToggle();
        setupPredictionDrag();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function onReady() {
            initializePredictionPanel();
        }, { once: true });
    } else {
        initializePredictionPanel();
    }
    document.addEventListener('shiny:connected', function() {
        initializePredictionPanel();
    });
})();
</script>
"""


def create_input_metadata():
    metadata = {}
    for col in categorical_cols_ui:
        values = sorted([str(v) for v in df[col].dropna().unique()])
        if values:
            metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}
    for col in numeric_cols_ui:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        vmin, vmax, vdef = float(s.min()), float(s.max()), float(s.median())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        step = max(1, round((vmax - vmin) / 200.0)) if (s.round() == s).all() else (vmax - vmin) / 200.0
        metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": vdef, "step": step}
    return metadata
input_metadata = create_input_metadata()

NUMERIC_SLIDER_IDS = [
    col
    for col in numeric_cols_ui
    if input_metadata.get(col, {}).get("type") == "numeric"
]

def create_widgets(cols, is_categorical=False):
    widgets = []
    for col in cols:
        if col not in input_metadata:
            continue
        meta = input_metadata[col]
        label = COLUMN_NAMES_KR.get(col, col)
        if is_categorical:
            widgets.append(
                ui.div(
                    ui.input_select(col, label, choices=meta["choices"], selected=meta["default"]),
                    class_="input-item",
                )
            )
        else:
            slider = ui.input_slider(
                col,
                label,
                min=meta["min"],
                max=meta["max"],
                value=meta["value"],
                step=meta["step"],
            )
            bar = ui.div(
                ui.div(id=f"{col}_normal_fill", class_="normal-range-fill"),
                id=f"{col}_normal_bar",
                class_="normal-range-bar",
            )
            label_div = ui.div(
                "",
                id=f"{col}_normal_label",
                class_="normal-range-label",
            )
            widgets.append(ui.div(slider, bar, label_div, class_="input-item"))
    return widgets
def panel_body():
    cat_widgets = create_widgets(categorical_cols_ui, True)
    num_widgets = create_widgets(numeric_cols_ui, False)

    cat_rows = [ui.layout_columns(*cat_widgets[i:i+4], col_widths=[3, 3, 3, 3]) for i in range(0, len(cat_widgets), 4)]
    num_rows = [ui.layout_columns(*num_widgets[i:i+4], col_widths=[3, 3, 3, 3]) for i in range(0, len(num_widgets), 4)]

    return ui.page_fluid(
        ui.HTML(custom_css + custom_js),
        ui.div(
            ui.div(
                ui.div("예측 결과", class_="card-header", style="background: #2A2D30; color: white; padding: 16px 20px; border-radius: 16px 16px 0 0; font-weight: 600; cursor: move;"),
                ui.div(
                    ui.div(ui.output_ui("prediction_result"), style="margin-bottom: 12px; text-align: center; font-size: 24px; font-weight: 700;"),
                    ui.div(ui.output_ui("probability_text"), style="margin-bottom: 12px; text-align: center; font-size: 14px; color: #495057;"),
                    ui.div(ui.output_ui("shap_top_features"), style="margin-bottom: 12px; font-size: 13px; color: #495057;"),
                    ui.div(ui.output_ui("recommendation_text"), style="margin-bottom: 16px; font-size: 13px; color: #495057;"),
                    ui.input_action_button("predict", "불량 여부 예측", style="width: 100%; background: #dc3545; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600; margin-bottom: 10px;"),
                    ui.input_action_button("apply_recommendations", "추천 정상값 적용", style="width: 100%; background: #9BE564; color: #2c3e50; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600; margin-bottom: 10px;"),
                    ui.input_action_button("load_defect_sample", "불량 샘플 랜덤 추출", style="width: 100%; background: #C2C0C0; color: #2c3e50; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600;"),
                    style="background: white; padding: 20px; border-radius: 0 0 16px 16px;"
                )
            ),
            id="draggable-prediction",
            style="position: fixed; bottom: 20px; right: 100px; width: 320px; z-index: 1000;"
        ),
        ui.HTML('<div id="settings-button" style="position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; background: #2A2D30; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.2); z-index: 1000; transition: transform 0.2s;"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg></div>'),
        ui.div(
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("변수 설정", style="font-size: 16px;"),
                        ui.span("접기", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAnalysisAccordion(event, 'variables_content')",
                    class_="accordion-header",
                    type="button"
                ),
                ui.div(*cat_rows, *num_rows, id="variables_content", class_="accordion-content", style="display: block;"),
                class_="accordion-section",
                style="margin-bottom: 16px;"
            ),
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("불량 샘플", style="font-size: 16px;"),
                        ui.span("접기", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAnalysisAccordion(event, 'defect_sample_content')",
                    class_="accordion-header",
                    type="button"
                ),
                ui.div(
                    ui.output_ui("defect_sample_table"),
                    ui.layout_columns(
                        ui.column(
                            12,
                            ui.div(
                                ui.output_plot("shap_force_plot", height="1010px"),
                                class_="shap-plot-card"
                            )
                        ),
                        ui.column(
                            12,
                            ui.div(
                                ui.output_plot("shap_force_overview_plot", height="480px"),
                                class_="shap-plot-card",
                                style="margin-bottom: 16px;"
                            ),
                            ui.div(
                                ui.output_plot("shap_waterfall_plot", height="480px"),
                                class_="shap-plot-card"
                            )
                        )
                    ),
                    id="defect_sample_content",
                    class_="accordion-content",
                    style="display: block;"
                ),
                class_="accordion-section"
            ),
            style="padding: 24px; max-width: 1900px; margin: 0 auto;"
        )
    )

def panel():
    return ui.nav_panel("예측 분석", panel_body())
def server(input, output, session):
    def _schedule_normal_range_update(ranges, features):
        payload = {
            "ranges": {k: v for k, v in ranges.items()},
            "features": list(features),
        }

        def _send():
            session.send_custom_message("update-normal-range", payload)

        session.on_flushed(_send, once=True)

    prediction_state = reactive.Value(None)
    active_sample = reactive.Value(None)
    show_defect_samples = reactive.Value(False)

    def _current_shap_explanation():
        details = prediction_state.get()
        if not details or details.get("error"):
            return None
        contributions = details.get("shap_aggregated")
        input_row = details.get("input_row")
        if not contributions or input_row is None:
            return None
        return build_shap_explanation(contributions, input_row)

    def _shap_placeholder(message="예측 결과를 먼저 계산하세요.", figsize=(6, 2.5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#6c757d")
        return fig

    @reactive.effect
    @reactive.event(input.load_defect_sample)
    def _load_samples():
        try:
            if "passorfail" not in df.columns:
                prediction_state.set({"error": "passorfail 컬럼이 없어 불량 샘플을 불러올 수 없습니다."})
                return
            defect = df[df["passorfail"] == 1].copy()
            if defect.empty:
                prediction_state.set({"error": "불량 샘플을 찾을 수 없습니다."})
                show_defect_samples.set(False)
                return
            sample_row = defect.sample(n=1).iloc[0]
            active_sample.set(sample_row)
            show_defect_samples.set(True)
            for col in all_cols:
                if col not in input_metadata:
                    continue
                val = sample_row.get(col)
                if pd.isna(val):
                    continue
                meta = input_metadata[col]
                if meta["type"] == "categorical":
                    value_str = str(val)
                    if value_str not in meta["choices"]:
                        continue
                    session.send_input_message(col, {"value": value_str})
                else:
                    numeric_val = float(val)
                    numeric_val = max(meta["min"], min(meta["max"], numeric_val))
                    session.send_input_message(col, {"value": numeric_val})
            prediction_state.set(None)
        except Exception as exc:
            prediction_state.set({"error": f"불량 샘플 적용 중 오류: {exc}"})

    @output
    @render.ui
    def defect_sample_table():
        if not show_defect_samples.get():
            return ui.div("불량 샘플 랜덤 추출 버튼을 눌러주세요.", style="text-align: center; color: #6c757d; padding: 20px;")
        sample_row = active_sample.get()
        if sample_row is None:
            return ui.div("불량 샘플 정보가 로드되지 않았습니다.", style="text-align: center; color: #6c757d; padding: 20px;")
        columns_to_display = [col for col in all_cols + ["passorfail"] if col in df.columns]
        row_df = pd.DataFrame([sample_row])[columns_to_display]
        html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px;"><thead style="background: #2A2D30; color: white;"><tr>'
        for col in row_df.columns:
            html += f'<th style="padding: 10px; text-align: left; border: 1px solid #dee2e6; white-space: nowrap;">{col}</th>'
        html += '</tr></thead><tbody><tr style="background: white;">'
        for col in row_df.columns:
            value = row_df.iloc[0][col]
            display_value = '-' if pd.isna(value) else str(value)
            html += f'<td style="padding: 8px; border: 1px solid #dee2e6; white-space: nowrap;">{display_value}</td>'
        html += '</tr>'
        html += '</tbody></table></div>'
        return ui.HTML(html)

    @reactive.effect
    @reactive.event(input.predict)
    def _run_prediction():
        try:
            input_row = {}
            for col in all_cols:
                meta = input_metadata.get(col)
                if not meta:
                    continue
                value = getattr(input, col)()
                if value is None:
                    continue
                if meta["type"] == "categorical":
                    input_row[col] = str(value)
                else:
                    input_row[col] = float(value)
            result = predict_with_model(input_row, compute_shap=True)
            aggregated = result.get("shap_aggregated", {})
            positive_items = [(feat, contrib) for feat, contrib in aggregated.items() if contrib > 0]
            if positive_items:
                positive_items.sort(key=lambda item: item[1], reverse=True)
                top_items = positive_items[:5]
            else:
                top_items = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
            top_features = []
            for feat, contrib in top_items:
                top_features.append({
                    "name": feat,
                    "label": COLUMN_NAMES_KR.get(feat, feat),
                    "value": input_row.get(feat, "-"),
                    "contribution": contrib
                })
            recommendations = recommend_ranges(input_row, [item["name"] for item in top_features])
            normal_range_payload = {}
            for feat, info in recommendations.items():
                if feat == "best_probability" or not isinstance(info, dict):
                    continue
                if info.get("type") != "numeric":
                    continue
                meta = input_metadata.get(feat, {})
                if not isinstance(meta, dict) or meta.get("type") != "numeric":
                    continue
                normal_min = info.get("min")
                normal_max = info.get("max")
                if normal_min is None or normal_max is None:
                    continue
                slider_min = meta.get("min")
                slider_max = meta.get("max")
                if slider_min is None or slider_max is None:
                    continue
                slider_min_f = float(slider_min)
                slider_max_f = float(slider_max)
                span = slider_max_f - slider_min_f
                if span <= 0:
                    continue
                normal_min_f = float(normal_min)
                normal_max_f = float(normal_max)
                start_pct = max(0.0, min(100.0, ((normal_min_f - slider_min_f) / span) * 100.0))
                end_pct = max(0.0, min(100.0, ((normal_max_f - slider_min_f) / span) * 100.0))
                if end_pct < start_pct:
                    start_pct, end_pct = end_pct, start_pct
                width_pct = max(0.0, end_pct - start_pct)
                fail_segments = []
                if start_pct > 0.0:
                    fail_segments.append(f"{slider_min_f:.1f}~{normal_min_f:.1f}")
                if end_pct < 100.0:
                    fail_segments.append(f"{normal_max_f:.1f}~{slider_max_f:.1f}")
                if fail_segments:
                    label_text = "불량 범위 약 " + ", ".join(fail_segments)
                else:
                    label_text = "불량 범위 없음"
                normal_range_payload[feat] = {
                    "normal_min": normal_min_f,
                    "normal_max": normal_max_f,
                    "slider_min": slider_min_f,
                    "slider_max": slider_max_f,
                    "start_pct": float(start_pct),
                    "width_pct": float(width_pct),
                    "label_text": label_text,
                }
            _schedule_normal_range_update(normal_range_payload, NUMERIC_SLIDER_IDS)
            result["top_features"] = top_features
            result["recommendations"] = recommendations
            result["input_row"] = input_row
            prediction_state.set(result)
        except Exception as exc:
            prediction_state.set({"error": str(exc)})
            _schedule_normal_range_update({}, NUMERIC_SLIDER_IDS)

    @reactive.effect
    @reactive.event(input.apply_recommendations)
    def _apply_recommendations():
        details = prediction_state.get()
        if not details or details.get("error"):
            return
        recommendations = details.get("recommendations") or {}
        if not recommendations:
            return
        target_features = []
        for item in details.get("top_features", []) or []:
            name = item.get("name")
            if name:
                target_features.append(name)
        if not target_features:
            target_features = [feat for feat in recommendations.keys() if feat != "best_probability"]
        updates = {}
        for feat in target_features:
            info = recommendations.get(feat)
            meta = input_metadata.get(feat)
            if not info or not meta:
                continue
            if meta.get("type") == "numeric":
                value = None
                for example in info.get("examples") or []:
                    try:
                        value = float(example)
                        break
                    except (TypeError, ValueError):
                        continue
                if value is None:
                    min_val = info.get("min")
                    max_val = info.get("max")
                    if min_val is not None and max_val is not None:
                        try:
                            value = (float(min_val) + float(max_val)) / 2.0
                        except (TypeError, ValueError):
                            value = None
                if value is None:
                    continue
                try:
                    min_bound = float(meta.get("min", value))
                    max_bound = float(meta.get("max", value))
                except (TypeError, ValueError):
                    min_bound = value
                    max_bound = value
                clipped = max(min_bound, min(max_bound, float(value)))
                updates[feat] = float(clipped)
            elif meta.get("type") == "categorical":
                values = info.get("values") or []
                if not values:
                    continue
                candidate = str(values[0])
                choices = meta.get("choices", [])
                if choices and candidate not in choices:
                    fallback = next((choice for choice in choices if choice in values), None)
                    if fallback is None:
                        continue
                    candidate = fallback
                updates[feat] = candidate
        if not updates:
            return
        for feat, value in updates.items():
            if isinstance(value, (int, float)):
                session.send_input_message(feat, {"value": float(value)})
            else:
                session.send_input_message(feat, {"value": str(value)})

    @output
    @render.ui
    def prediction_result():
        details = prediction_state.get()
        if not details:
            return ui.div("불량 여부 예측 버튼을 눌러 결과를 확인하세요.", style="font-size: 14px; font-weight: 500; color: #6c757d;")
        if "error" in details:
            return ui.div(f"오류 발생: {details['error']}", style="font-size: 16px; font-weight: 600; color: #dc3545;")
        label = "불량" if details["prediction"] == 1 else "정상"
        color = "#dc3545" if label == "불량" else "#28a745"
        note = " (tryshot_signal 규칙 적용)" if details.get("forced_fail") else ""
        return ui.div(f"{label}{note}", style=f"color: {color};")

    @output
    @render.ui
    def probability_text():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div("불량 확률은 예측 후 확인할 수 있습니다.")
        prob = details.get("probability", 0.0)
        text = f"불량 확률: {prob:.4f} (임계값 {operating_threshold:.4f})"
        if details.get("forced_fail"):
            text += " - 규칙에 의해 불량으로 판정됨"
        return ui.div(text)

    @output
    @render.ui
    def shap_top_features():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div()
        top_features = details.get("top_features", [])
        if not top_features:
            return ui.div("SHAP 상위 기여 요인을 확인하려면 예측을 실행하세요.")
        header_text = "불량 영향 상위 변수 (최대 5개)"
        items = []
        for item in top_features:
            value_text = format_value(item.get("value", "-"))
            contrib = item.get("contribution", 0.0)
            direction = "위험 증가" if contrib > 0 else "위험 감소"
            items.append(f"<li><strong>{item['label']}</strong> (현재 {value_text}) - SHAP {contrib:+.4f} ({direction})</li>")
        html = f"<div><span style='font-weight:600;'>{header_text}</span><ul style='padding-left:18px; margin:8px 0 0 0;'>{''.join(items)}</ul></div>"
        return ui.HTML(html)

    @output
    @render.plot
    def shap_force_plot():
        details = prediction_state.get()
        if not details or details.get("error"):
            return _shap_placeholder()
        focus_entries = details.get("top_features", []) or []
        focus_features = [item.get("name") for item in focus_entries if isinstance(item, dict) and item.get("name")]
        focus_features = [feat for feat in focus_features if feat in all_cols]
        if not focus_features:
            return _shap_placeholder("SHAP 상위 변수를 확인할 수 없습니다.")
        try:
            dist_data = compute_feature_distribution_data(focus_features)
            if not dist_data:
                return _shap_placeholder("분포 데이터를 계산할 수 없습니다.")
            shap_current = details.get("shap_aggregated", {}) or {}
            input_row = details.get("input_row") or {}
            recommendations = details.get("recommendations", {}) or {}
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_handles = [
                Patch(facecolor="#4dabf7", alpha=0.85, label="정상"),
                Patch(facecolor="#ff8787", alpha=0.85, label="불량"),
                Line2D([0, 1], [0, 0], color="#ff922b", linewidth=2, label="현재 값"),
            ]
            recommend_handle = Patch(facecolor="#e6fcf5", alpha=0.5, edgecolor="none", label="추천 범위")
            show_recommend = False
            fig_height = max(20, 6.5 * len(focus_features))
            fig, axes = plt.subplots(len(focus_features), 1, figsize=(11.5, fig_height), sharex=False)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.flatten()
            for ax, feat in zip(axes, focus_features):
                info = dist_data.get(feat)
                if not info:
                    ax.axis("off")
                    ax.text(0.5, 0.5, "데이터 없음", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#6c757d")
                    continue
                label = COLUMN_NAMES_KR.get(feat, feat)
                current_value = input_row.get(feat, "-")
                rec = recommendations.get(feat, {}) if isinstance(recommendations, dict) else {}
                ax.set_title(label, fontsize=9, pad=5)
                if info.get("type") == "numeric":
                    pass_vals = np.array(info.get("pass", []), dtype=float)
                    fail_vals = np.array(info.get("fail", []), dtype=float)
                    valid_pass = pass_vals[np.isfinite(pass_vals)]
                    valid_fail = fail_vals[np.isfinite(fail_vals)]
                    combined = np.concatenate([valid_pass, valid_fail]) if valid_pass.size + valid_fail.size > 0 else np.array([])
                    if combined.size == 0:
                        ax.axis("off")
                        ax.text(0.5, 0.5, "숫자 데이터 부족", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="#6c757d")
                        continue
                    bins = min(25, max(6, int(np.sqrt(combined.size))))
                    ax.hist(valid_pass, bins=bins, density=True, alpha=0.45, color="#4dabf7", edgecolor="white")
                    if valid_fail.size:
                        ax.hist(valid_fail, bins=bins, density=True, alpha=0.55, color="#ff8787", edgecolor="white")
                    try:
                        current_numeric = float(current_value)
                        ax.axvline(current_numeric, color="#ff922b", linewidth=2)
                    except (TypeError, ValueError):
                        pass
                    rec_min = rec.get("min")
                    rec_max = rec.get("max")
                    if rec_min is not None and rec_max is not None:
                        try:
                            rec_min = float(rec_min)
                            rec_max = float(rec_max)
                            if rec_max < rec_min:
                                rec_min, rec_max = rec_max, rec_min
                            ax.axvspan(rec_min, rec_max, color="#e6fcf5", alpha=0.45)
                            show_recommend = True
                        except (TypeError, ValueError):
                            pass
                    if combined.size > 0:
                        q_low, q_high = np.nanpercentile(combined, [1.5, 98.5])
                        span = q_high - q_low
                        margin = span * 0.15 if span > 0 else max(abs(q_low), abs(q_high), 1.0) * 0.15
                        ax.set_xlim(q_low - margin, q_high + margin)
                else:
                    pass_vals = info.get("pass", [])
                    fail_vals = info.get("fail", [])
                    categories = list(dict.fromkeys(pass_vals + fail_vals)) or ["missing"]
                    x = np.arange(len(categories))
                    pass_counts = pd.Series(pass_vals).value_counts(normalize=True)
                    fail_counts = pd.Series(fail_vals).value_counts(normalize=True) if fail_vals else pd.Series(dtype=float)
                    pass_heights = [pass_counts.get(cat, 0.0) for cat in categories]
                    fail_heights = [fail_counts.get(cat, 0.0) for cat in categories]
                    width = 0.35
                    ax.bar(x - width / 2, pass_heights, width=width, color="#4dabf7", alpha=0.85)
                    ax.bar(x + width / 2, fail_heights, width=width, color="#ff8787", alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(categories, rotation=0, fontsize=7)
                    ax.set_ylabel("비율", fontsize=9)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
                    ax.tick_params(axis="y", labelsize=8)
                    current_cat = str(current_value)
                    if current_cat in categories:
                        idx_cat = categories.index(current_cat)
                        marker_height = max(pass_heights[idx_cat], fail_heights[idx_cat]) + 0.03
                        ax.scatter([x[idx_cat]], [marker_height], marker="v", color="#ff922b", s=50, zorder=3)
                ax.grid(axis="y", linestyle=":", linewidth=0.4, color="#dee2e6", alpha=0.6)
            axes[-1].set_xlabel("값", fontsize=9)
            for ax in axes:
                ax.tick_params(axis="x", labelsize=8)
            handles = legend_handles.copy()
            if show_recommend:
                handles.append(recommend_handle)
            fig.subplots_adjust(top=0.9, bottom=0.06, hspace=0.65)
            fig.suptitle("상위 변수 값 분포 비교", fontsize=11, y=0.975)
            return fig
        except Exception as exc:
            return _shap_placeholder(f"분포 시각화 오류: {exc}")


    @output
    @output
    @render.plot
    def shap_force_overview_plot():
        details = prediction_state.get()
        if not details or details.get("error"):
            return _shap_placeholder("예측 결과를 먼저 계산하세요.", figsize=(7.0, 3.2))
        explanation = _current_shap_explanation()
        if explanation is None:
            return _shap_placeholder("예측 결과를 먼저 계산하세요.", figsize=(7.0, 3.2))
        try:
            values_all = np.array(explanation.values).reshape(-1)
            data_all = np.array(explanation.data).reshape(-1) if explanation.data is not None else None
            feature_names_all = list(explanation.feature_names or [])
            if not feature_names_all:
                feature_names_all = [f"feature_{idx}" for idx in range(len(values_all))]
            order = np.argsort(np.abs(values_all))[::-1]
            top_k = min(2, len(order))
            top_indices = order[:top_k]
            label_names = [COLUMN_NAMES_KR.get(feature_names_all[i], feature_names_all[i]) if i in top_indices else '' for i in range(len(feature_names_all))]
            plt.close('all')
            shap.force_plot(
                explanation.base_values,
                explanation.values,
                explanation.data,
                feature_names=label_names,
                matplotlib=True,
                show=False,
                text_rotation=0
            )
            fig = plt.gcf()
            fig.set_size_inches(7.0, 3.2)
            fig.tight_layout()
            fig.suptitle('상위 2개 변수 Force Plot', fontsize=11, y=0.98)
            ax = fig.axes[0] if fig.axes else None
            if ax is not None:
                for text_obj in list(ax.texts):
                    idx = getattr(text_obj, 'text_label', None)
                    txt = text_obj.get_text()
                    if idx is not None and idx in top_indices and txt:
                        if '=' in txt:
                            text_obj.set_text(txt.split('=')[0].strip())
                    else:
                        text_obj.set_alpha(0.0)
                line_count = 0
                for line in list(ax.lines):
                    idx = getattr(line, 'text_label', None)
                    if idx is not None:
                        if idx in top_indices and line_count < top_k:
                            line_count += 1
                        else:
                            line.set_alpha(0.0)
            return fig
        except Exception:
            return _shap_placeholder("SHAP force plot을 생성하는 중 문제가 발생했습니다.", figsize=(7.0, 3.2))
    @output
    @render.plot
    def shap_waterfall_plot():
        explanation = _current_shap_explanation()
        if explanation is None:
            return _shap_placeholder(figsize=(7.4, 5.2))
        try:
            values = np.array(explanation.values).reshape(-1)
            feature_names = list(explanation.feature_names or [])
            if not feature_names:
                feature_names = [f"feature_{idx}" for idx in range(len(values))]
            data_values = np.array(explanation.data).reshape(-1) if explanation.data is not None else None
            order = np.argsort(np.abs(values))[::-1]
            top_k = min(8, len(order))
            order = order[:top_k]
            values = values[order]
            feature_names = [COLUMN_NAMES_KR.get(feature_names[idx], feature_names[idx]) for idx in order]
            if data_values is not None:
                data_values = data_values[order]
            base_value = float(np.array(explanation.base_values).reshape(-1)[0]) if explanation.base_values is not None else 0.0
            running = [base_value]
            for val in values:
                running.append(running[-1] + val)
            final_value = running[-1]
            fig, ax = plt.subplots(figsize=(7.4, 0.9 * (len(values) + 3)))
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            y_positions = np.arange(len(values) + 2)
            ax.barh(0, 0.001, left=base_value, color="#adb5bd", height=0.6)
            ax.text(base_value, 0, f"기준값 {base_value:.2f}", ha="left", va="center", fontsize=9, color="#495057")
            for idx, val in enumerate(values):
                start = running[idx]
                end = running[idx + 1]
                width = end - start
                left = min(start, end)
                color = "#ff6b6b" if width >= 0 else "#4dabf7"
                ax.barh(idx + 1, width=abs(width), left=left, height=0.65, color=color, alpha=0.9)
                ax.plot([start, end], [idx + 1, idx + 1], color="#495057", linewidth=0.4)
                marker_color = "#e03131" if width >= 0 else "#1c7ed6"
                ax.scatter([end], [idx + 1], color=marker_color, s=24, zorder=3)
                ax.text(end, idx + 1 + 0.18, f"{val:+.3f}", fontsize=8.5, ha="center", va="bottom", color="#212529")
            ax.barh(len(values) + 1, 0.001, left=final_value, color="#212529", height=0.6)
            ax.text(final_value, len(values) + 1, f"예측값 {final_value:.3f}", ha="left", va="center", fontsize=9, color="#212529")
            labels = ["기준값"] + feature_names + ["예측값"]
            ax.set_yticks(y_positions)
            ax.set_yticklabels(labels, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("기여도 (SHAP value)", fontsize=10)
            ax.axvline(0, color="#868e96", linewidth=0.8, linestyle="--", alpha=0.8)
            ax.grid(axis="x", linestyle=":", linewidth=0.4, color="#ced4da", alpha=0.85)
            ax.set_ylim(len(values) + 1.5, -0.5)
            ax.set_title("Waterfall", fontsize=12, pad=12, color="#212529")
            fig.subplots_adjust(left=0.32, right=0.97, top=0.94, bottom=0.08)
            return fig
        except Exception:
            return _shap_placeholder("SHAP waterfall plot을 생성하는 중 문제가 발생했습니다.", figsize=(7.4, 5.2))

        def fmt(value):
            return "-" if value is None else format_value(value)

        for status in statuses:
            within = status.get("within", False)
            color = "#28a745" if within else "#dc3545"
            background = "#e7f6ec" if within else "#fdecef"
            label = status.get("label", status.get("feature", ""))
            current_val = fmt(status.get("current"))
            min_val = fmt(status.get("min"))
            max_val = fmt(status.get("max"))
            body = (
                f"<div style='border-radius:10px; padding:10px 12px; margin-bottom:8px; background:{background}; border-left:4px solid {color};'>"
                f"<div style='font-weight:600; color:{color};'>{label}</div>"
                f"<div style='font-size:12px; color:#495057;'>현재 {current_val} / 추천 {min_val} ~ {max_val}</div>"
                "</div>"
            )
            items.append(body)
        html = "<div><span style='font-weight:600;'>정상 구간 상태</span>" + "".join(items) + "</div>"
        return ui.HTML(html)

    @output
    @render.ui
    def recommendation_text():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div()
        recommendations = details.get("recommendations", {})
        if not recommendations:
            return ui.div("추천 정상 구간을 찾지 못했습니다. 다른 변수 조합을 시도해보세요.")
        best_prob = recommendations.get("best_probability")
        lines = []
        for feat, info in recommendations.items():
            if feat == "best_probability":
                continue
            label = COLUMN_NAMES_KR.get(feat, feat)
            if info.get("type") == "numeric":
                min_val = format_value(info.get("min"))
                max_val = format_value(info.get("max"))
                examples = ", ".join(format_value(v) for v in info.get("examples", []))
                lines.append(f"<li><strong>{label}</strong>: 추천 정상 범위 {min_val} ~ {max_val} </li>")
            elif info.get("type") == "categorical":
                values = ", ".join(info.get("values", []))
                lines.append(f"<li><strong>{label}</strong>: 추천 정상 값 {values}</li>")
        summary = "<ul style='padding-left:18px; margin:8px 0 0 0;'>" + "".join(lines) + "</ul>"
        if best_prob is not None:
            summary += f"<div style='margin-top:8px;'>추천 조합 적용 시 예상 불량 확률 ~ {best_prob:.4f}</div>"
        return ui.HTML("<div><span style='font-weight:600;'>정상 전환 추천 구간</span>" + summary + "</div>")


app = App(panel, server)























