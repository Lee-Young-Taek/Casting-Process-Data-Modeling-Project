import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\train.csv")
test_df = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\test.csv")
submission_df = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\submit.csv")

train_df.info()
train_df.head()
train_df.isna().sum()

# 대부분이 결측치인 행 제거
train_df.drop(19327, inplace=True)

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop", "registration_time"], inplace=True)

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

'''
결측치 처리 (molten_volume)
'''
# 시간에 따른 molten_volume 
train_df["molten_volume"].unique()

train_df.loc[train_df["molten_volume"].isna(), :]
train_df.groupby("mold_code")["molten_volume"]

custom_colors = {
    8412 : '#2ca02c',
    8573 : '#ff7f0e',
    8600 : "#ff0e0e",
    8722 : "#ffd70e",
    8917 : '#2ca02c'
}

# 코드별 전자교반 시간
train_df.groupby(["mold_code", "EMS_operation_time"])["passorfail"].count()

# 코드별 형체력 
sns.histplot(data=train_df.loc[(train_df["physical_strength"]<10000) & (train_df["physical_strength"]>600), :], x='physical_strength', hue='mold_code', kde=True)

# 코드별 주조 압력
sns.histplot(data=train_df.loc[train_df["cast_pressure"]>300, :], x='cast_pressure', hue='mold_code', kde=True)
sns.histplot(data=train_df.loc[(train_df["cast_pressure"]>300) & train_df["mold_code"].isin([8573, 8600, 8722]), :], x='cast_pressure', hue='mold_code', kde=True)

# 코드별 냉각수 온도
sns.histplot(data=train_df.loc[train_df["Coolant_temperature"] < 150, :], x='Coolant_temperature', hue='mold_code', palette=custom_colors, kde=True)
sns.histplot(data=train_df.loc[(train_df["Coolant_temperature"] < 150) & (train_df["mold_code"]).isin([8573, 8600, 8722]), :], x='Coolant_temperature', hue='mold_code', kde=True)

# 코드별 설비 작동 사이클 시간
sns.histplot(data=train_df.loc[(train_df["facility_operation_cycleTime"]<150) & (train_df["facility_operation_cycleTime"]>80), :], x='facility_operation_cycleTime', hue='mold_code', palette=custom_colors, kde=True)


train_df.isna().sum()

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

train_df["heating_furnace"]

'''
결측치 처리 (upper_mold_temp3)
버리기!
'''
train_df["upper_mold_temp3"].unique()
train_df["upper_mold_temp3"].isna().sum()
train_df[train_df["upper_mold_temp3"].isna()&train_df["lower_mold_temp3"].isna()]
train_df[train_df["upper_mold_temp3"].isna()].groupby(["mold_code"])["passorfail"].count()
train_df[train_df["lower_mold_temp3"].isna()].groupby(["mold_code"])["passorfail"].count()
train_df[train_df["upper_mold_temp3"].isna()].groupby(["mold_code"])["passorfail"].count()

#mold_code가 8412인 것 
train_df.loc[train_df["mold_code"] == 8412, "upper_mold_temp3"].unique()

train_df.loc[train_df["mold_code"] == 8412, "upper_mold_temp3">1000].mean()

train_df.loc[train_df["upper_mold_temp3"]< 1000, "passorfail"].value_counts()

train_df[train_df["upper_mold_temp3"].isna()].groupby(["mold_code"])["passorfail"].count()


'''
결측치 처리 (lower_mold_temp3)
버리기
'''
train_df[train_df["lower_mold_temp3"].isna()].groupby(["mold_code"])["passorfail"].count()

train_df.loc[train_df["lower_mold_temp3"]> 1400, "passorfail"].value_counts()
train_df.loc[train_df["lower_mold_temp3"]> 1400, "mold_code"].count()


'''
결측치 처리 (heating_furnace)
'''
train_df.loc[train_df["heating_furnace"].isna(),"mold_code"].value_counts()
train_df[train_df["heating_furnace"].isna()].groupby(["mold_code","passorfail"])["lower_mold_temp1"].count()
train_df["heating_furnace"].isna().sum()
train_df.loc[train_df["heating_furnace"].isna(),"mold_code"].value_counts()
train_df.loc[:,"mold_code"].value_counts()

train_df["mold_code"] / 

train_df["mold_code"] == ["a"]

train_df[train_df['mold_code'] == 'a']['lower_mold_temp3'].notna().sum()
train_df[train_df['mold_code'] == 'b']['lower_mold_temp3'].notna().sum()
