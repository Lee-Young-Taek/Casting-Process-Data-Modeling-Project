import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
SCALER_FILE = BASE_DIR / "data" / "interim" / "std_scaler_v1.joblib"
MODEL_FILE = BASE_DIR / "data" / "interim" / "rf_model_v1.joblib"
df = pd.read_csv(DATA_FILE, encoding='utf-8', low_memory=False)

# 데이터 불러오기
train_df = pd.read_csv("../../data/raw/train.csv")
test_df = pd.read_csv("../../data/raw/test.csv")

# 데이터 기본 정보 확인
train_df.info()
train_df.head()
train_df.isna().sum()

# 타켓 컬럼
target_col = "passorfail"

# 한번에 너무 많은 결측치가 있는 행 제거
train_df.drop(19327, inplace=True)

# 시험 운행 데이터 제거
train_df = train_df[~(train_df["tryshot_signal"] == "D")]

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop", "registration_time", "tryshot_signal"], inplace=True)

# 정수형인데 범주형으로 처리할 컬럼
train_df["mold_code"] = train_df["mold_code"].astype('object')

# 수치형 & 범주형 컬럼 리스트
num_cols = train_df.select_dtypes(include=["number"]).columns.tolist()
num_cols.remove(target_col)
cat_cols = train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# 모든 피처 리스트
all_features = [c for c in train_df.columns if c not in target_col]

# 결과 저장용 DataFrame
res = pd.DataFrame(index=all_features, columns=[
    'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'mutual_info', 'cramers_v'
], dtype=float)

# 수치형
y = train_df[target_col].astype(float)

for col in num_cols:
    x = train_df[col]
    common = x.notna() & y.notna()

    # point-biserial (y=이진, x=연속) - SciPy의 pointbiserialr는 (binary, continuous) 순서로 권장
    try:
        r_pb, p_pb = stats.pointbiserialr(y[common], x[common])
    except Exception:
        # 예외 시 pearson 대체
        r_pb, p_pb = stats.pearsonr(x[common], y[common])
    
    # Spearman
    r_sp, p_sp = stats.spearmanr(x[common], y[common])

    res.loc[col, ['pearson_r','pearson_p','spearman_r','spearman_p']] = [round(r_pb, 2), round(p_pb, 2), round(r_sp, 2), round(p_sp, 2)]


# 범주형
# Cramer's V 함수
def cramers_v(x, y):
    """
    두 범주형 시리즈 간 Cramer's V 계산.
    """
    # NaN을 문자열 '<<NA>>'로 대체하여 하나의 범주로 처리
    x = x.fillna('<<NA>>').astype(str)
    y = y.fillna('<<NA>>').astype(str)

    # 혼동 행렬 생성
    confusion = pd.crosstab(x, y)

    # 카이제곱 통계량 계산
    chi2, p, dof, ex = stats.chi2_contingency(confusion)
    
    # 표본 크기
    n = confusion.sum().sum()

    # Cramer's V 공식
    phi2 = chi2 / n
    
    # 행과 열의 수
    r, k = confusion.shape
    
    # bias 보정(작은 표본에 대한 보정)
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    denom = min(kcorr-1, rcorr-1)
    if denom <= 0:
        return np.nan
    return np.sqrt(phi2corr / denom)

for col in cat_cols:
    res.loc[col, 'cramers_v'] = round(cramers_v(train_df[col], train_df[target_col]), 2)

#  Mutual Information 계산 (모든 피처)
# 범주형은 라벨 인코딩, 숫자형은 중앙값으로 결측 대체
X = pd.DataFrame(index=train_df.index)
discrete_mask = []  # mutual_info_classif에 사용할 discrete mask

for col in all_features:
    if col in cat_cols:
        # 범주형: 라벨 인코딩, 결측은 '<<NA>>'로 대체
        le = LabelEncoder()
        tmp = train_df[col].fillna('<<NA>>').astype(str)
        X[col] = le.fit_transform(tmp)
        discrete_mask.append(True)
    else:
        # 숫자형: 결측은 중앙값으로 대체(단순 방법)
        if train_df[col].notna().sum() > 0:
            X[col] = train_df[col].astype(float).fillna(train_df[col].median())
        else:
            X[col] = train_df[col].astype(float).fillna(0.0)
        discrete_mask.append(False)

# mutual information 계산 (예외 처리 포함)
try:
    mi = mutual_info_classif(X.values, y.values, discrete_features=np.array(discrete_mask), random_state=2025)
    res['mutual_info'] = mi
except Exception as e:
    # 실패시 모든 변수를 연속변수로 처리해 시도
    try:
        mi = mutual_info_classif(X.fillna(0).values, y.values, random_state=2025)
        res['mutual_info'] = mi
    except Exception:
        res['mutual_info'] = np.nan

# 결과 출력
print("\n[숫자형 특성: Pearson 상관(절대값 기준) 상위 항목]")
top_pearson = res['pearson_r'].abs().sort_values(ascending=False).dropna().head(5)
print(top_pearson)

print("\n[숫자형 특성: Spearman 상관(절대값 기준) 상위 항목]")
top_spearman = res['spearman_r'].abs().sort_values(ascending=False).dropna().head(5)
print(top_spearman)

print("\n[Mutual Information 기준 상위 항목]")
top_mi = res['mutual_info'].sort_values(ascending=False).dropna().head(5)
print(top_mi)

print("\n[범주형 특성: Cramer's V 기준 상위 항목]")
top_cramer = res['cramers_v'].sort_values(ascending=False).dropna().head(5)
print(top_cramer)

# 결과 저장
res.to_csv("../../data/interim/target_association_summary.csv", index=True, encoding='utf-8-sig')