### KneeSense代码使用说明

#### 1.dataprocessed文件夹

- ##### 1.运行Two IMU + EMG.py接受IMU的蓝牙数据，数据默认收录在一级目录的dataset中

- ##### 2.运行resample.py进行补帧操作

  - 1将文件的绝对路径复制到代码42行中

  ```python
  imu_file = pd.read_csv(r'D:\Code_Project\Python\Knee_Sense\dataset\20241220_092613_d2.csv')
  ```

  - 2将文件名字复制到代码58行

    ```python
    # 构建完整的文件路径
    file_path = os.path.join(output_directory, '20241220_092613_d2.csv')
    ```

- ##### 3.运行CSV_process.py，输入文件时间戳，将IMU和EMG的数据分离

  - IMU1，IMU2, EMG文件自动保存在data_processed文件中

#### 2.angle_calculate文件夹

- ##### 1.将IMU_1和IMU_2的文件转录到imu1_data.csv,imu2_data.csv文件里

- ##### 2.运行individualtest.py,将csv文件转化为txt文件，默认保存到result1中

- ##### 3.运行JointAngle.py，生成角度文件，默认保存到result2中

  - 误差注意：由于IMU内置xyz轴设置的原因，要提前进行数据测试，可能会用到以下的代码

    ```python
    cur["angle_acc_gyr"] = angle_acc_gyr
    ```

    ```python
    cur["angle_acc_gyr"] = 180 - angle_acc_gyr
    ```

#### 3.txt_process文件夹

- ##### 1.将result2文件中生成的txt文件，运行cleandata.py进行数据清洗

  - 修改数据名称

  ```python
  fix_and_process_data('4.txt','4_process.txt')
  ```

- ##### 2.将清洗好的数据名称做好修改，运行filetransform.py进行格式修改

  ```python
  convert_format('4_process.txt', '4_final.txt')
  ```

#### 4.Visualize文件夹

- ##### 运行Visualize.py文件，将文件名字复制到文件中

  ```python
  file_path = r"20241220_092613_EMG.csv"
  ```

  

