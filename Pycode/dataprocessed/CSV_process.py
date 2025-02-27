"""
读取csv文档里的数据
d1分为IMU数据和 EMG
d2只读取IMU
"""
import os


def set_new_file(relative_file_name):
    # 构建完整的源文件路径
    dataset_path = os.path.join(os.getcwd(), 'dataset')
    source_file_path = os.path.join(dataset_path, relative_file_name)

    # 确保源文件存在
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"Source file '{relative_file_name}' not found in 'dataset' folder.")

    # 读取文件内容
    with open(source_file_path, 'r') as file:
        lines = file.readlines()

    # 检查IMU类型（这里需要根据您的具体文件名或内容来确定IMU类型）
    IMU_type = check_IMU(relative_file_name)

    # 构建处理后的文件保存路径
    dataset_processed_path = os.path.join(os.getcwd(), 'dataset_processed')
    if not os.path.exists(dataset_processed_path):
        os.makedirs(dataset_processed_path)

    # 获取列名
    column_name = lines[0].strip().split(',')
    IMU_column = '\t'.join(column_name[:7]) + '\n'
    EMG_column = column_name[-1] + '\n'

    # 根据IMU类型处理文件
    if IMU_type == 1:
        # 构建IMU和EMG文件路径
        imu_name = 'IMU' + str(IMU_type)
        imu_file_name = relative_file_name.replace('d1.csv', f'{imu_name}.csv')
        imu_file_path = os.path.join(dataset_processed_path, imu_file_name)

        emg_file_name = relative_file_name.replace('d1.csv', 'EMG.csv')
        emg_file_path = os.path.join(dataset_processed_path, emg_file_name)

        # 写入IMU和EMG文件头
        with open(imu_file_path, 'a') as IMU_file:
            if os.path.getsize(imu_file_path) == 0:
                IMU_file.write(IMU_column)

        with open(emg_file_path, 'a') as EMG_file:
            if os.path.getsize(emg_file_path) == 0:
                EMG_file.write(EMG_column)

        # 写入数据行
        for line in lines[1:]:
            context = line.strip().split(',')
            IMU_data = '\t'.join(context[:-1]) + '\n'
            with open(imu_file_path, 'a') as IMU_file:
                IMU_file.write(IMU_data)

            EMG_data = context[-1] + '\n'
            with open(emg_file_path, 'a') as EMG_file:
                EMG_file.write(EMG_data)
    else:
        # IMU类型不为1时的处理逻辑
        imu_name = 'IMU' + str(IMU_type)
        imu_file_name = relative_file_name.replace('d2.csv', f'{imu_name}.csv')
        imu_file_path = os.path.join(dataset_processed_path, imu_file_name)

        # 写入IMU文件头
        with open(imu_file_path, 'a') as IMU_file:
            if os.path.getsize(imu_file_path) == 0:
                IMU_file.write(IMU_column)

        # 写入数据行
        for line in lines[1:]:
            context = line.strip().split(',')
            IMU_data = '\t'.join(context[:-1]) + '\n'
            with open(imu_file_path, 'a') as IMU_file:
                IMU_file.write(IMU_data)



def check_IMU(file_name):
    name_parts = file_name.split('_')
    if 'd1.csv' in name_parts[-1]:
        IMU_type = 1
    else:
        IMU_type = 2
    return IMU_type


def main():
    # file_name_1 = "20241217_171759_d1.csv"
    # file_name_2 = "20241217_171759_d2.csv"
    time_1 = input("请输入当前文件的时间戳：")
    file_name_1 = time_1 + "_d1.csv"
    file_name_2 = time_1 + "_d2.csv"
    set_new_file(file_name_2)
    set_new_file(file_name_1)


if __name__ == "__main__":
    main()