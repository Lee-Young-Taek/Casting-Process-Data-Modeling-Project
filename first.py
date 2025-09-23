import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 불러오기
# train에는 없고 test에는 있는 값들이 있음
df = pd.read_csv("train.csv")

df.info()
# time / date/ count/ working/ emrgency_stop/molten_tmp / heating_furnace

################################################################################
# EDA
####################################################################################

# time 
# 년월일
# 결측치 확인
df['time'].isnull().sum()

# time 컬럼을 object > datetime 형식으로 변환
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d', errors='coerce')

# 변환 결과 확인
print(df['time'].head())

# 시간순으로 정렬
type(df['registration_time'][0]) # str > datetime
# registration_time을 datetime으로 변환
df['registration_time'] = pd.to_datetime(df['registration_time'], format="%Y-%m-%d %H:%M:%S", errors="coerce")
type(df['registration_time'][0]) # str > datetime

# registration_time 기준으로 정렬
df = df.sort_values(by="registration_time").reset_index(drop=True)

# 날짜 별 총 생산량 차이
daily_counts = df[['id','time']].groupby(['time']).count()

len(daily_counts)
# 시각화
plt.figure(figsize=(12,6))
plt.plot(daily_counts.index, daily_counts['id'], marker='o', linestyle='-')

plt.title("일자별 데이터 개수", fontsize=14)
plt.xlabel("날짜", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.grid(True)
plt.show()

# 날짜 별 제품 별 생산량 차이

# mold_code 별 일자별 개수 집계
daily_counts_by_mold = df.groupby(['time', 'mold_code'])['id'].count().reset_index()
daily_counts_by_mold[daily_counts_by_mold['mold_code']==8573]
# 피벗으로 변환 (행: time, 열: mold_code, 값: count)
pivot_counts = daily_counts_by_mold.pivot(index='time', columns='mold_code', values='id').fillna(0)

# 시각화
plt.figure(figsize=(12,6))

for col in pivot_counts.columns:
    plt.plot(pivot_counts.index, pivot_counts[col], marker='o', linestyle='-', label=f"mold_code {col}")

plt.title("일자별 mold_code별 생산량", fontsize=14)
plt.xlabel("날짜", fontsize=12)
plt.ylabel("개수", fontsize=12)
plt.legend(title="Mold Code")
plt.grid(True)
plt.show()

df[df['mold_code']==8573]


########################################################################################
# date ( time과 다르게 시간 분 초)
########################################################################################

# object > time 데이터 타입 변경
df['date'].dtypes
df['date'] = pd.to_datetime(df['date'], format='%H:%M:%S').dt.time

# 결측치 확인
df['time'].isnull().sum()

# 시간별 생산량 확인
# 2️⃣ 시간만 뽑아서 hour 컬럼 생성
df['hour'] = pd.to_datetime(df['date'].astype(str)).dt.hour

# 3️⃣ 시간별 생산량 집계
hourly_counts = df.groupby('hour').size()

# 4️⃣ 시각화
plt.figure(figsize=(12,6))
hourly_counts.plot(kind='bar')
plt.title("시간대별 생산량")
plt.xlabel("시간 (시)")
plt.ylabel("생산량")
plt.xticks(rotation=0)
plt.show()

########################################################################################
# count
########################################################################################

# 결측치 확인
df['count'].isnull().sum()
df_8412 = df[df['mold_code']==8412].reset_index()
df_8412[['time','date','count']].head(30)
da
# 날짜별 count 최대값
# count가 초기화되는 기준이 다름
# 위에서 id로 count()함수 사용해서 하루 생산량 구한 거랑 일별 count의 max값이 다르기 떄문
daily_max = df_8412.groupby('time')['count'].max().reset_index()

df_8412[df_8412['time'] == '2019-01-02']['count']

# count가 최소값인 행 필터링
min_count_rows = df[df['count'] == df['count'].min()]

# 시간대별 분포 확인
hour_distribution = min_count_rows['hour'].value_counts().sort_index()

# 출력
print(hour_distribution)

# 시각화 (선택)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
hour_distribution.plot(kind='bar')
plt.title("count 최소값 행의 시간대별 분포")
plt.xlabel("시간 (시)")
plt.ylabel("행 수")
plt.xticks(rotation=0)
plt.show()

# 제품별 최소 count 발생 시간대 확인
min_count_hours = df[df['count'] == df['count'].min()].groupby('mold_code')['hour'].unique()

# 결과 확인
print(min_count_hours)

import matplotlib.pyplot as plt
import seaborn as sns

# 제품별 최소 count 행 추출
min_count_rows = df[df['count'] == df['count'].min()]

# 시각화
plt.figure(figsize=(12,6))
sns.countplot(data=min_count_rows, x='hour', hue='mold_code', palette='Set2')
plt.title("제품별 count 최소값 발생 시간대")
plt.xlabel("시간 (시)")
plt.ylabel("행 수")
plt.legend(title='Mold Code')
plt.xticks(rotation=0)
plt.show()


# count가 07~08시 그리고 19~20시에 초기화됨


# 제품별 count 최대값
# 제품별, 날짜별 count 최대값 구하기
df['date_only'] = pd.to_datetime(df['time'].dt.date)  # 'time' 컬럼이 datetime이면 이렇게 가능

daily_max = df.groupby(['mold_code', 'date_only'])['count'].max().reset_index(name='daily_max')

avg_daily_max_by_product = daily_max.groupby('mold_code')['daily_max'].mean()
print(avg_daily_max_by_product)



####################################################################################
# working
####################################################################################
df.info()
# 결측치 확인
df['working'].isnull().sum() # 1개 > 드랍

# 상태별 건수 집계 # 정지 38개
status_counts = df['working'].value_counts()

# 막대그래프 시각화
plt.figure(figsize=(6,4))
status_counts.plot(kind='bar', color=['green','red'])
plt.title("설비 상태별 생산량")
plt.xlabel("상태")
plt.ylabel("건수")
plt.xticks(rotation=0)
plt.show()

# 정지 총 38개 > 2개는 정상
df[df['working']=='정지']['passorfail'].value_counts()

####################################################################################
# emrgency_stop
####################################################################################

# 결측치 확인
df['emergency_stop'].isnull().sum() # 1개 드랍

# 분포 확인
df['emergency_stop'].value_counts() # ON 값 1개만 있어서 없애도 되는 열인 듯?

# 결측치 행 확인
df[df['emergency_stop'].isna()]

####################################################################################
#molten_temp
####################################################################################

df[df['mold_code']==8722]['count']
# 결측치 2261개
df['molten_temp'].isnull().sum()


import pandas as pd

# molten_code별로 그룹핑해서 앞뒤 평균으로 결측치 채우기
df['molten_temp_filled'] = (
    df.groupby('mold_code')['molten_temp']
      .apply(lambda x: x.fillna((x.ffill() + x.bfill()) / 2))
)

# 혹시 남아 있는 결측치는 다시 보간 처리 (예방용)
df['molten_temp_filled'] = (
    df.groupby('mold_code')['molten_temp_filled']
      .apply(lambda x: x.interpolate(method='linear'))
)
# 히스토그렘
plt.figure(figsize=(10,6))
df['molten_temp'].hist(bins=100)  # bin 수를 50으로 늘림
plt.xlabel("용탕 온도")
plt.ylabel("건수")
plt.title("용탕 온도 분포")
plt.show()

df['molten_temp'].describe()

# 이상치로 의심되는 데이터 개수
df[df['molten_temp'] < 600] # 434 개
df[df['molten_temp'] < 100]['passorfail'].value_counts()

# 의심되는 애들 제거하고 새로 시각화
df = df[df['molten_temp'] >= 600]

# 히스토그렘
plt.figure(figsize=(10,6))
df['molten_temp'].hist(bins=100)  # bin 수를 50으로 늘림
plt.xlabel("용탕 온도")
plt.ylabel("건수")
plt.title("용탕 온도 분포")
plt.show()

# 불량 양품 비교
plt.figure(figsize=(10,6))
sns.boxplot(x='passorfail', y='molten_temp', data=df)
plt.xticks([0,1], ['양품', '불량'])
plt.xlabel("품질")
plt.ylabel("용탕 온도")
plt.title("불량/양품에 따른 용탕 온도 분포")
plt.show()

# 구간 별 불량률 계산해야할듯
df[df['tryshot_signal']=="D"]
####################################################################################
# heating_furnace
####################################################################################

# 결측치 38208개 
df['heating_furnace'].isnull().sum()

# 값 확인
df['heating_furnace'].value_counts()

# 결측치에 'nan' 문자열로 채우기
df['heating_furnace'] = df['heating_furnace'].fillna('nan')

# 상태별 개수 집계
furnace_counts = df['heating_furnace'].value_counts()

# 막대그래프 시각화
plt.figure(figsize=(6,4))
furnace_counts.plot(kind='bar', color=['orange','blue','gray'])
plt.title("가열로 구분별 샘플 수 (결측치 'nan' 포함)")
plt.xlabel("가열로")
plt.ylabel("건수")
plt.xticks(rotation=0)
plt.show()
df
df['mold_name'].value_counts()


###############################################################
train_df = df.copy()
# 일자(time), 제품(mold_code)별 count 중복 횟수 확인
dup_stats = (
    train_df
    .groupby(['time','mold_code'])['count']
    .value_counts()   # count별 등장 횟수
    .reset_index(name='dup_freq')  # 컬럼 이름 정리
    .sort_values(['dup_freq'], ascending=False)  # 중복 많은 순 정렬
)
train_df.groupby('time')['count'].value_counts()


train_df[train_df['mold_code']==8412]['count']
train_df[train_df['mold_code']==8573]['count']
train_df[train_df['mold_code']==8412]['count']
train_df[train_df['mold_code']==8412]['count']
df.info()
train_df['mold_code'].unique()



import pandas as pd
import matplotlib.pyplot as plt

mold_codes = [8722, 8412, 8573, 8917, 8600]
bin_size = 50  # 구간 단위

plt.figure(figsize=(12,8))

for code in mold_codes:
    df_sub = train_df[train_df['mold_code'] == code].copy()
    df_sub['count_bin'] = (df_sub['count'] // bin_size) * bin_size
    
    defect_stats = (
        df_sub.groupby('count_bin')
        .agg(total=('count', 'size'),
             defects=('passorfail', 'sum'))
    )
    defect_stats['defect_rate'] = defect_stats['defects'] / defect_stats['total']
    
    plt.plot(defect_stats.index, defect_stats['defect_rate'], marker='o', label=f"mold_code {code}")

plt.xlabel("Count 구간 시작")
plt.ylabel("불량률")
plt.title("Mold_code별 구간별 불량률")
plt.legend()
plt.grid(True)
plt.show()
