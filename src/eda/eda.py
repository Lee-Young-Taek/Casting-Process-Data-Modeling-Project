import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

# 데이터 로드
train_df = pd.read_csv(DATA_FILE)

# 데이터 정보
train_df.info()
train_df.columns

# date, time 컬럼명 swap
train_df = train_df.rename(columns={'date': '__tmp_swap__'})
train_df = train_df.rename(columns={'time': 'date', '__tmp_swap__': 'time'})

# date, time 컬럼 타입 변환
train_df["date"] = pd.to_datetime(train_df["date"], format="%Y-%m-%d")
train_df["time"] = pd.to_datetime(train_df["time"], format="%H:%M:%S")

# 시간별 생산량
# 확인결과 7시, 19시는 생산량이 매우 적음
df = train_df.copy()
df["hour"] = df["time"].dt.hour
by_hour = df.groupby('hour').size().reindex(range(24), fill_value=0)

ax = by_hour.plot(kind='bar', figsize=(10,4))
ax.set_xlabel('Hour of day (0-23)')
ax.set_ylabel('Count (rows)')
ax.set_title('Production Count by Hour')
plt.tight_layout()
plt.show()

# 데이터 결측치 확인하기
train_df.isna().sum()

# 대부분이 결측치인 행 확인
train_df.iloc[19327, :]
mold_code_19327 = train_df.loc[19327, "mold_code"]
time_19327 = train_df.loc[19327, "time"]
train_df.loc[(train_df["mold_code"] == mold_code_19327) & (train_df["time"] == time_19327) & (train_df["id"] > 19273), :]
# 해당 행이 유일한 emergency_stop 결측행이여서 이 행이 긴급중단을 나타내는 행인지 판단

# 데이터 컬럼별 유니크값 수
train_df.nunique().sort_values()

# 해당 컬럼들은 단일값 컬럼 이므로 제거
train_df["line"].unique()
train_df["name"].unique()
train_df["mold_name"].unique()
train_df.drop(columns=["line", "name", "mold_name"])

# 해당 컬럼들은 nan값이 하나의 값으로 나타날 것 같아 제거 하지 않음.
train_df["emergency_stop"].unique()
train_df["tryshot_signal"].unique()

# mold_code별로 데이터 프레임 나누기
mold_codes = train_df["mold_code"].unique()
df_8722 = train_df[train_df["mold_code"] == 8722].copy()
df_8412 = train_df[train_df["mold_code"] == 8412].copy()
df_8573 = train_df[train_df["mold_code"] == 8573].copy()
df_8917 = train_df[train_df["mold_code"] == 8917].copy()
df_8600 = train_df[train_df["mold_code"] == 8600].copy()

# 연속된 count 행 제거 함수
def remove_consecutive_counts(df):
    prev_count = 0
    index_list = []

    for idx, row in df.iterrows():
        if row["count"] == prev_count:
            index_list.append(idx)
        prev_count = row["count"]

    df.drop(index=index_list, inplace=True)
    return df

df_8722 = remove_consecutive_counts(df_8722)
df_8412 = remove_consecutive_counts(df_8412)
df_8573 = remove_consecutive_counts(df_8573)
df_8917 = remove_consecutive_counts(df_8917)
df_8600 = remove_consecutive_counts(df_8600)

