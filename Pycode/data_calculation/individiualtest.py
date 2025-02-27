import numpy as np
from scipy import interpolate
import datetime
import os


# def get_time(timestr):
#     timestr = timestr.strip()
#     cur = datetime.datetime.strptime(timestr, '%Y-%m-%d %H:%M:%S.%f')
#     return cur.timestamp()
def get_time(timestr):
    # 直接返回浮点数形式的 Unix 时间戳
    return float(timestr.strip())

def get_raw_data(paths=['imu1_data.csv', 'imu2_data.csv']):
    data = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        with open(path, 'r') as f:
            # 假设数据是用制表符分隔的，而不是逗号
            context = f.read().split('\n')
            if len(context) < 2:
                raise ValueError(f"The file {path} does not contain enough data.")
            # 通常CSV文件的第一行是列名，这里我们不需要它，但可以记录下来以供参考
            # names = context[0].split('\t')  # 如果需要列名的话
            data_rows = [row.split('\t') for row in context[1:]]  # 使用制表符分隔
            device_data = []
            for row in data_rows:
                if row and row[0]:  # 确保行不为空且有时间戳
                    # 只将时间戳部分转换为浮点数
                    timestamp = get_time(row[0])
                    # 将剩余部分也转换为浮点数列表
                    values = [float(row[i]) for i in range(1, len(row))]
                    # 组合时间戳和值列表
                    device_data.append([timestamp] + values)
            data.append(device_data)
    return data
# def get_raw_data(paths=['imu1_data.csv', 'imu2_data.csv']):
#     data = []
#     for path in paths:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"The file {path} does not exist.")
#         with open(path) as f:
#             context = f.read().split('\n')
#             if len(context) < 2:
#                 raise ValueError(f"The file {path} does not contain enough data.")
#             names = context[0].split(',')
#             data_rows = [row.split(',') for row in context[1:]]
#             device_data = []
#             for row in data_rows:
#                 if row and row[0]:
#                     cur = [get_time(row[0])] + [float(row[i]) for i in range(1, 7)]
#                     device_data.append(cur)
#             data.append(device_data)
#     return data


def get_inter(data, begin, end, diff):
    tot = begin
    whole = []
    nx = []
    while tot < end:
        whole.append([tot])
        nx.append(tot)
        tot = round(tot + diff, 6)

    x = [item[0] for item in data]
    for i in range(1, len(data[0])):
        cur = [row[i] for row in data]
        f = interpolate.interp1d(x, cur, kind='linear')
        y = f(nx)
        for j in range(len(whole)):
            whole[j].append(float(y[j]))

    return whole


def get_angular_acceleration(data, diff, index):
    vel_dot = []
    for i in range(len(data)):
        cur = []
        for j in index:
            res = 0.0
            if i > 1 and i < len(data) - 2:
                res = (data[i - 2][j] - 8 * data[i - 1][j] + 8 * data[i + 1][j] - data[i + 2][j]) / (12 * diff)
            elif i == 1 or i == len(data) - 2:
                res = (data[i + 1][j] - data[i - 1][j]) / (2 * diff)
            elif i == 0:
                res = (data[i + 1][j] - data[i][j]) / diff
            else:  # Should be else if i == len(data) - 1
                res = (data[i][j] - data[i - 1][j]) / diff
            cur.append(float('%.6f' % res))
        vel_dot.append(cur)
    for i in range(len(data)):
        data[i] += vel_dot[i]
    return data


def main(number, paths=['imu1_data.csv', 'imu2_data.csv']):
    if not os.path.exists('result1'):
        os.makedirs('result1')

    raw_data = get_raw_data(paths)
    imu1 = get_inter(raw_data[0], raw_data[0][0][0], raw_data[0][-1][0], 0.001)
    imu2 = get_inter(raw_data[1], raw_data[1][0][0], raw_data[1][-1][0], 0.001)

    with open('result1/imu1_%d.txt' % number, 'w') as file1:
        for i in imu1:
            file1.write('\t'.join(['%.6f' % x for x in i]) + '\n')

    with open('result1/imu2_%d.txt' % number, 'w') as file2:
        for i in imu2:
            file2.write('\t'.join(['%.6f' % x for x in i]) + '\n')


# Run the main function with an example number
main(2)