import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
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

train_df.info()
train_df.head()
train_df.isna().sum()

# 대부분이 결측치인 행 제거
train_df.drop(19327, inplace=True)

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop", "registration_time"], inplace=True)

# tryshot == "D" 행 제거
train_df = train_df.loc[~(train_df["tryshot_signal"] == "D"), :]
train_df.drop(columns=["tryshot_signal"], inplace=True)

'''
결측치 처리 (molten_temp)
동일코드 앞 생산 온도, 동일 코드 뒤 생산 온도 평균
'''
# 🔹 원본 molten_temp를 새로운 열로 복사
train_df['molten_temp_filled'] = train_df['molten_temp']

# 🔹 금형별 시간 순 정렬 후 선형 보간
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 🔹 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)
train_df[['molten_temp', 'molten_temp_filled']]
train_df['molten_temp'].isna().sum()
train_df['molten_temp_filled'].isna().sum()
train_df.drop(columns=["molten_temp"], inplace=True)

# 수치형 -> 범주형
train_df["mold_code"] = train_df["mold_code"].astype('object')
train_df["EMS_operation_time"] = train_df["EMS_operation_time"].astype("object")

# heating_furnace 결측치 처리
train_df["heating_furnace"].fillna("c", inplace=True)

# date, time dt 바꾸기
train_df['date'] = pd.to_datetime(train_df['date'], format='%H:%M:%S')
train_df['time'] = pd.to_datetime(train_df['time'], format='%Y-%m-%d', errors='coerce')

# 시간 컬럼 만들기
train_df["hour"] = train_df["date"].dt.hour
train_df.drop(columns=["date"], inplace=True)

# 요일 컬럼 만들기
train_df["weekday"] = train_df["time"].dt.weekday
train_df.drop(columns=["time"], inplace=True)

# 수치형, 범주형 컬럼 선택
num_columns = train_df[["cast_pressure", "count", "upper_mold_temp1", "low_section_speed", "lower_mold_temp2", 
                       "high_section_speed", "upper_mold_temp2", "lower_mold_temp1", "biscuit_thickness", "sleeve_temperature"]].columns
cat_columns = train_df.select_dtypes(include=['object']).columns

# 결측치 채우기 (간단히 처리)
freq_impute = SimpleImputer(strategy='most_frequent')
mean_impute = SimpleImputer(strategy='mean')

train_df[cat_columns] = freq_impute.fit_transform(train_df[cat_columns])
train_df[num_columns] = mean_impute.fit_transform(train_df[num_columns])

# 인코딩, 스케일링
onehot = OneHotEncoder(handle_unknown='ignore',                        
                       sparse_output=False).set_output(transform="pandas")
std_scaler = StandardScaler().set_output(transform="pandas")

train_df_cat = onehot.fit_transform(train_df[cat_columns])
train_df_num = std_scaler.fit_transform(train_df[num_columns])

train_df_all = pd.concat([train_df_cat,
                          train_df_num], axis = 1)

# X, y 설정
X_train = train_df_all[["cast_pressure", "count", "upper_mold_temp1", "low_section_speed", "lower_mold_temp2", 
                       "high_section_speed", "upper_mold_temp2", "lower_mold_temp1", "biscuit_thickness", "sleeve_temperature"]]
y_train = train_df["passorfail"]

rf = RandomForestClassifier(oob_score=True)

rf.fit(X_train, y_train)

test_X = X_train.head(1)
test_X["cast_pressure"] = "1.5"

pred = rf.predict(test_X)[0]

joblib.dump(rf, "../../data/interim/rf_model_v1.joblib")
joblib.dump(std_scaler, "../../data/interim/std_scaler_v1.joblib")
