import pandas as pd
import numpy as np
import scipy.interpolate
import os


def resample_imu_data(imu_file, desired_sample_rate=30):
    """
    Resample with timestamps array
    :param imu_file: IMU数据文件,list
    :param desired_sample_rate: 重采样的频率
    """
    num_columns = imu_file.shape[1]
    timestamp = imu_file["ts_receiver"]
    time = timestamp.to_numpy() - timestamp.iloc[0]
    resampled_imu_file = pd.DataFrame(columns=range(num_columns))
    for i in range(num_columns - 1):
        interp_func = scipy.interpolate.interp1d(time, imu_file.iloc[:, i + 1].to_numpy(), kind='linear')
        sampling_interval = 1.0 / desired_sample_rate
        new_timestamps = np.arange(time[0], time[-1], sampling_interval)  # 6.2
        resampled_imu_file[i + 1] = interp_func(new_timestamps)
        new_timestamps = new_timestamps + timestamp.iloc[0]
    resampled_imu_file[0] = new_timestamps
    return resampled_imu_file

# Example initialization of imu_file DataFrame

# data = {
#     "ts_receiver": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
#     "sensor_1": [1, 2, 3, 4, 5, 6]
# }
#
# data2 = {
#     "ts_receiver": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#     "sensor_2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# }
#
# imu_file = pd.DataFrame(data)
# imu_file2 = pd.DataFrame(data2)


imu_file = pd.read_csv(r'D:\Code_Project\Python\Knee_Sense\dataset\20241220_092613_d2.csv')

# Call the resample function
resampled_data = resample_imu_data(imu_file, desired_sample_rate=20)
# resampled_data2 = resample_imu_data(imu_file2, desired_sample_rate=10)
# print(resampled_data)
# print(resampled_data2)

resampled_data.columns = ['ts_receiver', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'EMG']

# 确保dataset_processed文件夹存在
output_directory = 'dataset'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 构建完整的文件路径
file_path = os.path.join(output_directory, '20241220_092613_d2.csv')

# 将resampled_data保存到CSV文件中
resampled_data.to_csv(file_path, index=False)

# resampled_data.to_csv('20241217_215305_d2.csv', index=False)