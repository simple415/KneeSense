import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r"20241220_092613_EMG.csv"
data = pd.read_csv(file_path, header=None, skiprows=1) 

# 获取EMG数据
emg_data = pd.to_numeric(data[0], errors='coerce')

# 清理NaN值
emg_data = emg_data.dropna()
time = [i * (1/20) for i in range(len(emg_data))]

# 绘制原始EMG数据
plt.plot(time, emg_data)
plt.xlabel('Time (s)')
plt.ylabel('EMG Value (mV)')
plt.title('EMG Real-Time Data')
plt.grid(True)

plt.show()
