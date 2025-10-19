# 데이터와 데이터셋 불러오기
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.reset_option('display.max_rows')
df = pd.read_csv('차량 연비 데이터(2002~2022).csv')

#df데이터 정보보기 (컬럼,데이터타입,행의 개수, null값 확인)
df.info()

#null의 개수 파악
df.isnull().sum()

#중복된 행 개수 파악 -> 1개가 파악됨
df.duplicated().sum()

#중복된 행 제거
df.drop_duplicates(inplace = True)

#컬럼 공백제거, 소문자 및 공백은 언더바('_')로 변환 -> 코드작성하는데 편리함을 위해
df.columns = df.columns.str.strip()
df.columns = df.columns.str.lower().str.replace(' ','_')

# 원본데이터는 유지하고 작업할 데이터 따로 생성 
df1 = df.copy()

#make에 대문자 통일
df['make']= df['make'].str.upper()

#vehicle 대문자 통일
df["vehicle_class"] = df["vehicle_class"].str.upper()

#실린더수 4,6,8인 데이터만 추출
df = df[(df['cylinders']==4) | (df['cylinders'] == 6) | (df['cylinders'] == 8)]

def classify_vehicle(x):
    x = x.upper().strip()
    small = ['MINICOMPACT', 'SUBCOMPACT', 'COMPACT', 'SUV - SMALL', 'SUV: SMALL',
             'STATION WAGON - SMALL', 'STATION WAGON: SMALL',
             'PICKUP TRUCK - SMALL', 'PICKUP TRUCK: SMALL', 'TWO-SEATER']

    mid = ['MID-SIZE', 'STATION WAGON - MID-SIZE', 'STATION WAGON: MID-SIZE',
           'SUV', 'SUV - STANDARD', 'SUV: STANDARD',
           'PICKUP TRUCK - STANDARD', 'PICKUP TRUCK: STANDARD', 'MINIVAN']

    full = ['FULL-SIZE', 'VAN - PASSENGER', 'VAN: PASSENGER', 'VAN - CARGO',
            'SPECIAL PURPOSE VEHICLE']

    if x in full:
        return 'full'
    elif x in mid:
        return 'mid'
    elif x in small:
        return 'small'
df1['VEHICLE CLASS'] = df['VEHICLE CLASS'].apply(classify_vehicle)
df1

# 소형,중형,대형으로 새로운 데이터 프레임 생성
small_df = df1[df1['VEHICLE CLASS']=='small']
mid_df = df1[df1['VEHICLE CLASS']=='mid']
full_df = df1[df1['VEHICLE CLASS'] == 'full']

#이상치 확인
def iqr_int(small_df,column_name):
    #1사분위
    q1 = small_df[column_name].quantile(0.25) 
    #3사분위
    q3 = small_df[column_name].quantile(0.75) 
    #IQR 범위
    iqr = q3 - q1

    #수염 공식
    lower_bound = q1 - 1.5*iqr 
    upper_bound = q3 + 1.5*iqr

    #Q1, Q3, IQR, 수염 소수점 두자리까지 표현
    print(f"Q1 : {q1:.2f}")
    print(f"Q3 : {q3:.2f}")
    print(f"IAR : {iqr:.2f}")
    print(f"하향선 : {lower_bound:.2f}")
    print(f"상향선 : {upper_bound:.2f}")

    #<컬럼의 각 값이 lower_bound보다 작은 것 |(또는) upper_bound보다 큰 것을> df에서 찾아서 필터링 
    # series는 or대신 | 를 써야 한다.
    outliar_lower = small_df[(small_df[column_name]<lower_bound)]
    outliar_upper = small_df[(small_df[column_name]>upper_bound)]
    outliar_iqr = small_df[(small_df[column_name]<lower_bound) | (small_df[column_name]>upper_bound)] 
    
    #print(f"{outliar_lower[column_name]}")
    print(f"{len(outliar_lower)}")
    #print(f"{outliar_upper[column_name]}")
    print(f"{len(outliar_upper)}")
    #print(f"{outliar_iqr[column_name]}")
    print(f"탐지된 이상치 개수 :{len(outliar_iqr)}개")

    #outliar에 비어있지 않으면 즉, 하나라도 있으면 참 -> 해당 컬럼에 이상치가 있으면 출력
    if not outliar_iqr.empty:
        print (f"{column_name} 이상치 상세:")
    else:
        print(f"IQR기준으로 {column_name} 이상치가 발견되지 않았습니다.")

    #새로움 그림(figure)객체 생성 (가로:8인치, 세로:6인치)
    plt.figure(figsize=(8,6))
    #seaborn으로 박스 플롯 생성, 해당컬럼의 요소가 y축에 표시되도록하고, 색 지정
    sns.boxplot(y=small_df[column_name], color = 'skyblue')
    #stripplot은 "개별 데이터 포인트(이상치)"를 시각화, 해당컬럼의 이상치의 요소값을 y축에 표시, 색 지정, 포인트 사이즈
    #jitter는 데이터 포인트가 겹치지 않도록 False로 설정하여 정확한 위치 표시, marker로 포인트 모양 지정, 범례 레이블 지정
    sns.stripplot(y=outliar_iqr[column_name], color='red', size=8, jitter=False, marker='o', label='outliar')
    #그래프 제목 설정
    plt.title(f'"{column_name} IQR', fontsize=16)
    #y축의 라벨 설정
    plt.ylabel(f'{column_name} (Value)', fontsize=12)
    #label -> sns.stripplot에서 outliar에 대한 범례 표시
    plt.legend(fontsize=10)
    #격자 추가, axis=y는 y축에만, 선의 스타일은 점섬으로, 투명도는70%로 설정)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#각 컬럼별 이상치 확인(중형, 대형은 _df앞 값만 변경하여 확인)
iqr_int(small_df,'ENGINE SIZE')
iqr_int(small_df,'CYLINDERS')
iqr_int(small_df,'CITY (L/100_km)')
iqr_int(small_df,'HWY (L/100_km)')
iqr_int(small_df,'COMB (L/100_km)')
iqr_int(small_df,'COMB (mpg)')
iqr_int(small_df,'EMISSIONS')

# 이상치 제거 함수
def remove_IQR(df1, columns):
    Q1 = df1[columns].quantile(0.25)
    Q3 = df1[columns].quantile(0.75)
    IQR = Q3- Q1
    min = Q1 - 1.5*IQR
    max = Q3 + 1.5*IQR
    return df1[(df1[columns]>min) & (df1[columns]<max)]

# 소형 이상치 제거
remove_small = remove_IQR(small_df, 'ENGINE SIZE')
small_final = remove_IQR(remove_small, 'CYLINDERS')

# 중형 이상치 제거
mid_final = remove_IQR(mid_df,'ENGINE SIZE')

# 대형 이상치 제거
full_final = remove_IQR(full_df, 'CYLINDERS')

# 이상치 제거한 전체 데이터 통합
final_df1 = pd.concat([small_final, mid_final, full_final])


# 9페이지
# 전체 연비와 배출량과 상관관계
sns.lmplot(x='COMB (L/100_km)', y='EMISSIONS', data=small_df, height=6, aspect=1.2)
plt.title('연비 vs 배출량')
plt.show()


# 10페이지
# 차량 크기별 연비와 배출량 상관관계
final_df1.groupby('VEHICLE CLASS')['EMISSIONS'].mean().round(2)

# 🔹 원하는 차량 순서 정의
vehicle_order = ['small', 'mid', 'full']

# 🔹 카테고리형으로 순서 지정
final_df1['VEHICLE CLASS'] = pd.Categorical(df['VEHICLE CLASS'], categories=vehicle_order, ordered=True)

# 평균값 계산
avg_comb = df.groupby('VEHICLE CLASS')['COMB (L/100 km)'].mean().round(2)
avg_emission = df.groupby('VEHICLE CLASS')['EMISSIONS'].mean().round(2)

# 시각화
fig, ax1 = plt.subplots()

# 연비 막대그래프 (연한 연두색)
ax1.bar(avg_comb.index, avg_comb.values, color='#d9ead3')
ax1.set_ylabel('연비 (L/100km)', color='#d9ead3')
ax1.set_ylim(15, 8)
ax1.invert_yaxis()

# 배출량 선그래프 (짙은 파랑)
ax2 = ax1.twinx()
ax2.plot(avg_emission.index, avg_emission.values, color='#003366', marker='o')
ax2.set_ylabel('배출량 (g/km)', color='#003366')

# 제목
plt.title('차량 크기별 평균 연비와 이산화탄소 배출량 비교', fontsize=14, weight='bold')
plt.grid(True)
plt.show()

# 11페이지
#소형 연비와 배출량 상관관계
df_encoded = small_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['CITY_(L/100_KM)','HWY_(L/100_KM)','COMB_(MPG)','COMB_(L/100_KM)','EMISSIONS']
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(9,7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#중형 연비와 배출량 상관관계
df_encoded = mid_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['CITY_(L/100_KM)','HWY_(L/100_KM)','COMB_(MPG)','COMB_(L/100_KM)','EMISSIONS']
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(9,7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#대형 연비와 배출량 상관관계
df_encoded = full_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['CITY_(L/100_KM)','HWY_(L/100_KM)','COMB_(MPG)','COMB_(L/100_KM)','EMISSIONS']
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(9,7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

# 12페이지
# 구동방식 범주화 A, AM, AV, M, A

transmission_types = []

for t in final_df1['TRANSMISSION']:
    if t.startswith('AS'):
        transmission_types.append('AS')
    elif t.startswith('AM'):
        transmission_types.append('AM')
    elif t.startswith('AV'):
        transmission_types.append('AV')
    elif t.startswith('M'):
        transmission_types.append('M')
    elif t.startswith('A'):
        transmission_types.append('A')
    else:
        transmission_types.append('Unknown')

final_df1['TRANSMISSION_TYPE'] = transmission_types

# 크기별, 구동방식별 배출랭 평균(낮은순으로) 추출
grouped = df.groupby(['SIZE', 'TRANSMISSION_TYPE'])['EMISSIONS'].mean().round(1).reset_index()
grouped = grouped.sort_values(by=['SIZE', 'EMISSIONS'], ascending=[False, True]).reset_index(drop=True)

# display(grouped)

# 위 추출내용 시각화
pivot_mpg = df.pivot_table(index='TRANSMISSION_TYPE', columns='SIZE', values='EMISSIONS', aggfunc='mean')
sns.heatmap(pivot_mpg, annot=True, cmap='Greens', fmt=".0f")
plt.title('차량 크기별 · 구동방식별 평균 배출량')
plt.show()

# 13페이지
#소형 구동방식 히트맵
transmission_types = []
# 카테고리 생성
for t in small_final['transmission']:
    if t.startswith('AS'):
        transmission_types.append('AS')
    elif t.startswith('AM'):
        transmission_types.append('AM')
    elif t.startswith('AV'):
        transmission_types.append('AV')
    elif t.startswith('M'):
        transmission_types.append('M')
    elif t.startswith('A'):
        transmission_types.append('A')
    else:
        transmission_types.append('Unknown')
small_final['TRANS'] = transmission_types

small_trans = small_final
df_encoded = pd.get_dummies(small_trans, columns=['TRANS'], drop_first = False)
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('TRANS_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#중형 구동방식 히트맵
transmission_types = []
# 카테고리 생성
for t in mid_final['transmission']:
    if t.startswith('AS'):
        transmission_types.append('AS')
    elif t.startswith('AM'):
        transmission_types.append('AM')
    elif t.startswith('AV'):
        transmission_types.append('AV')
    elif t.startswith('M'):
        transmission_types.append('M')
    elif t.startswith('A'):
        transmission_types.append('A')
    else:
        transmission_types.append('Unknown')
mid_final['TRANS'] = transmission_types

mid_trans = mid_final
df_encoded = pd.get_dummies(mid_trans, columns=['TRANS'], drop_first = False)
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('TRANS_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#대형 구동방식 히트맵
transmission_types = []
# 카테고리 생성
for t in full_final['transmission']:
    if t.startswith('AS'):
        transmission_types.append('AS')
    elif t.startswith('AM'):
        transmission_types.append('AM')
    elif t.startswith('AV'):
        transmission_types.append('AV')
    elif t.startswith('M'):
        transmission_types.append('M')
    elif t.startswith('A'):
        transmission_types.append('A')
    else:
        transmission_types.append('Unknown')
full_final['TRANS'] = transmission_types

full_trans = full_final
df_encoded = pd.get_dummies(full_trans, columns=['TRANS'], drop_first = False)
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('TRANS_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

# 14페이지
# 소형차량 연료별/배출량 카테고리 생성
small_fuel = small_final.groupby('FUEL')['EMISSIONS'].mean().reset_index()
# 소형차량 연료별/배출량 막대그래프로 시각화
plt.figure(figsize=(8, 5))
plt.bar(small_fuel['FUEL'], small_fuel['EMISSIONS'], color='green')
plt.xlabel('FUEL TYPE')
plt.ylabel('Average EMISSIONS')
plt.title('Average Emissions by Fuel Type(small)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 중형차량 연료별/배출량 카테고리 생성
mid_fuel = mid_final.groupby('FUEL')['EMISSIONS'].mean().reset_index()
# 중형차량 연료별/배출량 막대그래프로 시각화
plt.figure(figsize=(8, 5))
plt.bar(mid_fuel['FUEL'], mid_fuel['EMISSIONS'], color='blue')
plt.xlabel('FUEL TYPE')
plt.ylabel('Average EMISSIONS')
plt.title('Average Emissions by Fuel Type(mid)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
# 대형차량 연료별/배출량 카테고리 생성
full_fuel = full_final.groupby('FUEL')['EMISSIONS'].mean().reset_index()
# 대형차량 연료별/배출량 막대그래프로 시각화
plt.figure(figsize=(8, 5))
plt.bar(full_fuel['FUEL'], full_fuel['EMISSIONS'], color='orange')
plt.xlabel('FUEL TYPE')
plt.ylabel('Average EMISSIONS')
plt.title('Average Emissions by Fuel Type(full)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 15페이지
#소형 연료별 배출량 상관관계
df_encoded = small_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
df_encoded = pd.get_dummies(df_encoded, columns=['FUEL'], drop_first = False)
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('FUEL_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#중형 연료별 배출량 상관관계
df_encoded = mid_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
df_encoded = pd.get_dummies(df_encoded, columns=['FUEL'], drop_first = False)
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('FUEL_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()

#대형 연료별 배출량 상관관계
df_encoded = full_final
df_encoded.columns = df_encoded.columns.str.upper().str.replace(' ','_')
df_encoded = pd.get_dummies(df_encoded, columns=['FUEL'], drop_first = False)
relevant_cols = ['EMISSIONS']
for col in df_encoded.columns:
    if col.startswith('FUEL_'):
        relevant_cols.append(col)
correlation_matrix = df_encoded[relevant_cols].corr()
plt.Figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('CORRELATION', fontsize=14)
plt.show()


# 16페이지
#window 사용자용 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#소형
engine_emission_small = small_final.groupby(['ENGINE SIZE'])['EMISSIONS'].mean().reset_index().sort_values('EMISSIONS', ascending=True).reset_index(drop=True)
bins_small = [0, 2.0, 4.0, 6.0]
labels_small = ['size1','size2','size3']
engine_emission_small['engine_size_category'] = pd.cut(engine_emission_small['ENGINE SIZE'], bins=bins_small, labels=labels_small, right=False)
avg_emissions_small = engine_emission_small.groupby('engine_size_category')['EMISSIONS'].mean().reset_index()
avg_emissions_small['car_type'] = '소형' 

#중형
engine_emi_mid = mid_final.groupby(['ENGINE SIZE'])['EMISSIONS'].mean().reset_index().sort_values('EMISSIONS', ascending=True).reset_index(drop=True)
bins_mid = [0, 2.0, 4.0, 6.0, 8.0]
labels_mid = ['size1','size2','size3','size4']
engine_emi_mid['engine_size_category'] = pd.cut(engine_emi_mid['ENGINE SIZE'], bins=bins_mid, labels=labels_mid, right=False)
avg_emissions_mid = engine_emi_mid.groupby('engine_size_category')['EMISSIONS'].mean().reset_index()
avg_emissions_mid['car_type'] = '중형' 

#대형
engine_emi_full = full_final.groupby(['ENGINE SIZE'])['EMISSIONS'].mean().reset_index().sort_values('EMISSIONS', ascending=True).reset_index(drop=True)
bins_full = [0, 2.0, 4.0, 6.0, 8.0]
labels_full = ['size1','size2','size3','size4']
engine_emi_full['engine_size_category'] = pd.cut(engine_emi_full['ENGINE SIZE'], bins=bins_full, labels=labels_full, right=False)
avg_emissions_full = engine_emi_full.groupby('engine_size_category')['EMISSIONS'].mean().reset_index()
avg_emissions_full['car_type'] = '대형' 

all_avg_emissions = pd.concat([avg_emissions_small, avg_emissions_mid, avg_emissions_full])

plt.figure(figsize=(10, 6))

sns.lineplot(data=all_avg_emissions, x='engine_size_category', y='EMISSIONS', hue='car_type', marker='o')            
        
plt.title('엔진 크기별 평균 배출량 (소형, 중형, 대형 비교)') 
plt.xlabel('엔진 크기 분류') 
plt.ylabel('평균 배출량')    
plt.grid(True)               
plt.legend(title='차량 종류') 
plt.show()

#17페이지
##### 소형
small_final[small_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)


# 소형 차량 중 실린더별 평균 배출량 계산
small_emission = small_final[small_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)

# 시각화
plt.figure(figsize=(4, 5))
bars = plt.bar(small_emission.index.astype(str), small_emission.values, color='#d9ead3')
plt.xlabel('실린더 개수')
plt.ylabel('평균 이산화탄소 배출량 (g/km)')
plt.title('소형차 실린더 개수별 평균 이산화탄소 배출량', fontsize=13, weight='bold',pad=20)
plt.ylim(100, 350)

# ✅ 수치값 라벨 추가
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

##### 중형
mid_final[mid_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)

##### 중형 시각화
import matplotlib.pyplot as plt

# 중형 차량 중 실린더별 평균 배출량 계산
mid_emission = mid_final[mid_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)

# 시각화
plt.figure(figsize=(4, 5))  # 가로 폭 절반
bars = plt.bar(mid_emission.index.astype(str), mid_emission.values, color='#aace9aff')
plt.xlabel('실린더 개수')
plt.ylabel('평균 이산화탄소 배출량 (g/km)')
plt.title('중형차 실린더 개수별 평균 이산화탄소 배출량', fontsize=11, weight='bold', pad=20)
plt.ylim(100, 350)  # y축 범위 동일

# 수치값 라벨 추가
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


##### 대형
full_final[full_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)

##### 대형 시각화
import matplotlib.pyplot as plt

# 대형 차량 중 실린더별 평균 배출량 계산
full_emission = full_final[full_final['VEHICLE CLASS']].groupby('CYLINDERS')['EMISSIONS'].mean().round(2)

# 시각화
plt.figure(figsize=(4, 5))  # 가로 폭 절반
bars = plt.bar(full_emission.index.astype(str), full_emission.values, color='#91ac86ff')
plt.xlabel('실린더 개수')
plt.ylabel('평균 이산화탄소 배출량 (g/km)')
plt.title('대형차 실린더 개수별 평균 이산화탄소 배출량', fontsize=11, weight='bold', pad=20)
plt.ylim(100, 350)  # y축 범위 동일

# 수치값 라벨 추가
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f'{yval}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()


# 18페이지
# 소형차 복합연비 연도별 변화추이
small_final.groupby('YEAR')[['COMB (L/100 km)']].mean().plot(kind='line', marker='o')
# 중형차 복합연비 연도별 변화추이
mid_final.groupby('YEAR')[['COMB (L/100 km)']].mean().round(2).plot(kind='line', marker='o')
# 대형차 복합연비 연도별 변화추이
full_final.groupby('YEAR')[['COMB (L/100 km)']].mean().plot(kind='line', marker='o')

# 20페이지
# 소형차 엔진크기별 배출량 적은순 상위10개
small_final.groupby('ENGINE SIZE')['EMISSIONS'].mean().head(10)
# 중형차 엔진크기별 배출량 적은순 상위10개
mid_final.groupby('ENGINE SIZE')['EMISSIONS'].mean().head(10)
# 대형차 엔진크기별 배출량 적은순 상위10개
full_final.groupby('ENGINE SIZE')['EMISSIONS'].mean().head(10)

# 21페이지
# 소형차 실린더 개수별 배출량 적은순
small_final.groupby('CYLINDERS')['EMISSIONS'].mean()
# 중형차 실린더 개수별 배출량 적은순
mid_final.groupby('CYLINDERS')['EMISSIONS'].mean()
# 대형차 실린더 개수별 배출량 적은순
full_final.groupby('CYLINDERS')['EMISSIONS'].mean()

# 22페이지
# 소형차 연도별 복합연비 배출량 적은순 상위 10개
small_final.groupby('YEAR')[['COMB (L/100 km)', 'EMISSIONS']].mean().sort_values(by='EMISSIONS', ascending=True).head(10).round(2)
# 중형차 연도별 복합연비 배출량 적은순 상위 10개
mid_final.groupby('YEAR')[['COMB (L/100 km)', 'EMISSIONS']].mean().sort_values(by='EMISSIONS', ascending=True).head(10).round(2)
# 대형차 연도별 복합연비 배출량 적은순 상위 10개
full_final.groupby('YEAR')[['COMB (L/100 km)', 'EMISSIONS']].mean().sort_values(by='EMISSIONS', ascending=True).head(10).round(2)
