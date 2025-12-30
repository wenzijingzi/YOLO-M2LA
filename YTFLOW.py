import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ===================== 1. 数据读取 =====================
file_path = r"E:\doctorate\YTjiaotongpeishi\guanhailu_4_0901.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# ===================== 2. 时间转换 =====================
# 将 cxsj 转换为 datetime 格式
df['time'] = pd.to_datetime(df['cxsj'], format='%Y%m%d%H%M%S')

# ===================== 3. OD 匹配（简化示例） =====================
# 假设 sbmc 表示卡口位置，hphm 表示车牌哈希
df = df.sort_values(['hphm', 'time'])

# 匹配同一车辆的相邻卡口记录
df['next_sbmc'] = df.groupby('hphm')['sbmc'].shift(-1)
df['next_time'] = df.groupby('hphm')['time'].shift(-1)

# 计算旅行时间（秒）
df['travel_time'] = (df['next_time'] - df['time']).dt.total_seconds()

# ===================== 4. 路段平均速度与 TTI =====================
# 假设我们已知各路段长度（单位：米），可用字典存储
road_length = {
    "观海路与府后路交叉口（卡口）北1-观海路下一个卡口": 500,  # 示例 500m
    # 其他路段可依次填写
}

def compute_speed(row):
    key = f"{row['sbmc']}-{row['next_sbmc']}"
    if key in road_length and row['travel_time'] > 0:
        return road_length[key] / row['travel_time'] * 3.6  # 转换为 km/h
    return np.nan

df['speed'] = df.apply(compute_speed, axis=1)

# 旅行时间指数（TTI）
free_flow_speed = 40  # 自由流速度假设 40 km/h
df['TTI'] = free_flow_speed / df['speed']

# ===================== 5. 流量统计 =====================
df['hour'] = df['time'].dt.hour
traffic_volume = df.groupby(['sbmc', 'hour']).size().reset_index(name='volume')

# ===================== 6. 路口饱和度、拥堵指数 =====================
# 假设：饱和流率 s = 1800 pcu/h/ln，车道数 n=2，有效绿信比 g/C=0.5
s = 1800
n = 2
gC = 0.5
capacity = s * gC * n

# 计算饱和度
traffic_volume['v/c'] = traffic_volume['volume'] / capacity

# 拥堵指数 CI（示例权重）
alpha, beta, gamma = 0.4, 0.3, 0.3
df['CI'] = alpha * df['TTI'] + beta * (df['travel_time'] / 60) + gamma * (traffic_volume['v/c'].mean())

# ===================== 7. 可视化示例 =====================
plt.figure(figsize=(10,5))
traffic_volume.pivot(index='hour', columns='sbmc', values='volume').plot(kind='line', marker='o')
plt.title("各卡口分时交通流量")
plt.ylabel("车辆数（辆/小时）")
plt.xlabel("小时")
plt.grid(True)
plt.show()

# ===================== 8. 输出统计结果 =====================
# 路段平均速度、TTI
summary = df.groupby('sbmc').agg({
    'speed': 'mean',
    'TTI': 'mean',
    'travel_time': 'median'
}).reset_index()

print("=== 路段运行指标 ===")
print(summary)

print("=== 路口饱和度 ===")
print(traffic_volume[['sbmc', 'hour', 'volume', 'v/c']])
