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

import matplotlib.pyplot as plt

train_df.groupby('heating_furnace')['molten_temp'].mean().plot(kind='bar')
plt.title('가열로별 평균 용탕량')
plt.show()

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

# 결측치 처리
# molten_volume
train_df.groupby("mold_code")["molten_volume"].count()
train_df.loc[train_df["mold_code"] == 8412, "molten_volume"].hist()

'''
결측치 처리 (molten_volume)
8573 결측치는 8600, 8722로 회귀를 돌려서 채운다.
이유 : 전자교반 시간을 봤을 때 이 3가지는 같은 타입으로 유추 가능
나머지 2개 결측치는 코드별 평균을 채운다.

y로 몰드코드를 보는건가?
8573값이 너무 비어져있으니깐

knn을 돌린다?
비슷하다고 판단은 어떻게 알아? 그래프보고
knn으로
비슷하다는걸 어떻게 알아?
주변 이웃들 묶엇을 때 더 많은 쪽으로 예측한다?

형체력, 주조 입력, 냉각수 온도, 설비 작동 사이클 시간

x에 전자교반시간등등을 넣고?
'''

train_df.groupby("mold_code")["molten_volume"].size()
train_df.groupby("mold_code")["molten_volume"].count()

train_df.groupby("mold_code")["molten_volume"].size() - train_df.groupby("mold_code")["molten_volume"].count()
train_df["molten_temp"].mean()


'''
train_df.columns

#컬럼 설명
#준비과정
#heating_furnace : 가열로 (전자레인지)
train_df["heating_furnace"].unique()
-> A, B라는 가열로를 쓰는구나

#molten_temp : 용탕온도 (끓는 물 온도)
train_df["molten_temp"].unique()
-> 731도에서 알루미늄을 녹이는구나

#molten_volume : 용탕량 (녹은 알루미늄 리터)
train_df["molten_volume"].unique()
train_df["molten_volume"].mean().round(2)
->

#기계설정하기
#cast_pressure : 주조압력 (얼마나 세게 밀어넣을지)
train_df["cast_pressure"].unique()
-> 331만큼 압력을 준다?

#low_section_speed : 저속구간속도(처음엔 천천히)
train_df["low_section_speed"].unique()
-> 속도로 느리게 누른다?

#high_section_speed : 고속구간속도(나중엔 빠르게)
train_df["high_section_speed"].unique()
->112만큼 속도로 빠르게 누른다는건가?

#facility_operation_CycleTime : 설비사이클시간(압력이 가해지는 그걸 만드는데 걸리는 시간?)
train_df["facility_operation_cycleTime"].unique()
->119초/분? 정도로 시간이 걸린다?

#금형(틀) 준비하기
#mold_name : 금형명(어떤 틀을 사용할지)
train_df["mold_name"].unique()
->TM Carrier RH-Semi-Solid DIE-06금형틀을 사용하자

#upper_mold_temp1~3 : 상금형온도(윗부분 틀의 온도3군데)
train_df["upper_mold_temp1"].unique()
->164도정도다?

#lower_mold_temp1~3 : 하금형온도(아랫부분 틀의 온도3군데)
train_df["lower_mold_temp1"].unique()
->101도 정도다?

#sleeve_temperature : 슬리브 온도(주입구 온도)
train_df["sleeve_temperature"].unique()
-> 온도 452도다?

#실제 제작
#working : 가동여부(기계가 작동중인지)
train_df["working"].unique()
-> 가동/정지/nan

#emergency_stop : 비상정지(문제 발생 시 비상정지)
train_df["emergency_stop"].unique()
-> on/nan

#production_CycleTime : 생산시간(실제로 걸린시간)
train_df["production_cycletime"].unique()
-> 120분 정도 걸린다?

#냉각&마무리
#Coolant_temperature : 냉각수온도(찬물로 식히는 온도)
train_df["Coolant_temperature"].unique()
-> 34도 정도?

#biscuit_thickness : 비스캣두께(제품 두께 측정)
train_df["biscuit_thickness"].unique()
-> 35정도다

'''

