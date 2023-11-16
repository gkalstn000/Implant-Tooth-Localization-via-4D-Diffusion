import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/home/minsu/cad/results/diffusion_ct_nobatch_video_result/scores.csv')

positive = df[df.label == 1]
negative = df[df.label == 0]

def get_statics(df) :
    mean = df.score.mean()
    var = df.score.var()

    return mean, var

mean_p, var_p = get_statics(positive)
mean_n, var_n = get_statics(negative)


# 데이터프레임 로드 (예시)
# df = pd.read_csv('your_file.csv')

# 데이터프레임의 'score'와 'label' 열 시각화
# 'label' 별로 다른 색상으로 'score' 분포를 표시
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='score', data=df)
plt.title('Score Distribution by Label')
plt.xlabel('Label')
plt.ylabel('Score')
plt.show()

# 또는 'score' 분포의 히스토그램을 각 'label'별로 겹쳐서 표시할 수 있습니다.
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='score', hue='label', stat='density', common_norm=False)
plt.title('Score Distribution by Label')
plt.xlabel('Score')
plt.ylabel('Density')
plt.show()