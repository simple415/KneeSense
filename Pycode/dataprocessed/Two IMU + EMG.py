# -*- coding: utf-8 -*-

"""
device type: xiao_ble_sense
device name: IMU-Acc-gyro-1 DataLogger AND IMU-Acc-gyro-2 DataLogger AND IMU-Acc-gyro-3 DataLogger
Function: 连接xiao_ble_sense并进行数据接收
FIXME: 如果设备连接中，终止程序，而设备并没有通过程序进行正常断开，而是通过直接关闭蓝牙的方式断开，\\
    那么设备会发生错误，导致之后无法检索到设备，暂时通过端口识别可重新将设备恢复正常。
"""

import os
import time
import asyncio  # 异步并发
from aioconsole import ainput  # base asyncio, asynchronous https://aioconsole.readthedocs.io/en/latest/index.html

from datetime import datetime
from threading import *
from queue import *

from bleak import discover  # https://bleak.readthedocs.io/en/latest/index.html
from bleak import BleakClient
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation

warnings.filterwarnings("ignore")
# package_directory = os.path.dirname(os.path.abspath(__file__))

run_data_logging = False

# 本文件所在目录的路径
package_directory = os.path.dirname(os.path.abspath(__file__))

# 构建上一级目录的路径
parent_dir = os.path.dirname(package_directory)

# 文件存储的路径, user x
save_data_path = '../dataset'  # 相对路径


class BLEScanner:
    def __init__(self):
        self.ble_devices = {}
        self.run = True
        self.ready = False
        self.loop = asyncio.get_event_loop()
        self.ble_scanner_t = Thread(target=self.ble_scanner)
        self.ble_scanner_t.start()
        while not self.ready:
            time.sleep(0.1)

    async def find_device(self):
        devices = await discover()
        # print('the devices are ', devices)
        for d in devices:
            # print('the device id', d)
            if not check_key(self.ble_devices, d.name):
                # if not check_key_hy(self.ble_devices, d.name, d.address):
                self.ble_devices[d.name] = d.address
                print("New BLE device found: " + str(d.name) + " : " + str(d.address))

    def ble_scanner(self):
        while self.run:
            loop.run_until_complete(self.find_device())
            self.ready = True
            time.sleep(0.1)

    def close(self):
        self.run = False
        time.sleep(1)

    def get_address(self, device_name):
        # print('device_name', device_name)
        return check_key(self.ble_devices, device_name)


def check_key(_dict, key):
    if key in _dict.keys():
        # print('every device name',_dict[key])
        return _dict[key]
    else:
        return False


class Connection:
    client: BleakClient = None
    # acc_gyro_mag_char = "00e00000-0001-11e1-ac36-0002a5d5c51b"
    acc_gyro_mag_char = "af879017-8c9c-4092-8da1-0d115d08fa79"  # "64cf715e-f89e-4ec0-b5c5-d10ad9b53bf2"  # acc-x

    # quaternion_char = "00000100-0001-11e1-ac36-0002a5d5c51b"

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,  # 这是一个 asyncio 的事件循环对象，用于协调异步操作。在使用异步操作的情况下，通常需要提供一个事件循环对象。
                 device_address: str = None,
                 device_name: str = None,
                 data_q: Queue = None):

        self.loop = loop
        self.connected_device = device_address
        self.device_name = device_name  # hy
        self.data_q = data_q  # 这里转换成自己的队列了

        self.connected = False

        self.last_packet_time = datetime.now()
        self.rx_data = []
        self.rx_timestamps = []
        self.rx_delays = []

        # **jyc**# async def find_device(self):
        # **jyc**#     devices = await discover()
        # **jyc**#     print('the waiting names of this device',devices)
        # **jyc**#     for d in devices:
        # **jyc**#         if self.device_name == d.name:
        # **jyc**#             self.connected_device = d.address
        # **jyc**#             return
        # **jyc**#     if not self.connected_device:
        # **jyc**#         #print("Device: {} not found.".format(self.device_name))
        # **jyc**#         raise Exception

    def on_disconnect(self):
        self.connected = False
        # print("Disconnected!" + str(self.connected))

    def notification_handler(self, sender: str, data: any):
        self.rx_data.append(int.from_bytes(data, byteorder="big"))

    async def manager(self):
        print("Starting connection manager.")
        while True:
            if self.client:
                await self.connect()
            else:
                self.client = BleakClient(self.connected_device, loop=self.loop)
                await asyncio.sleep(1.0)

    async def connect(self):
        if self.connected:
            return
        try:
            self.client.set_disconnected_callback(self.on_disconnect())
            await self.client.connect()
            self.connected = await self.client.is_connected()
            if self.connected:
                print("{} connected!".format(self.connected_device))  # hy device_name
                await self.client.start_notify(self.acc_gyro_mag_char, self.data_callback)
                # await self.client.start_notify(self.quaternion_char, self.data_callback)

                while True:
                    if not self.connected:
                        break
                    await asyncio.sleep(5.0)

            else:
                print("Failed to connect.")

        except Exception as e:
            print(e)

    async def cleanup(self):
        if self.client:
            await self.client.stop_notify(self.acc_gyro_mag_char)
            # await self.client.stop_notify(self.quaternion_char)
            await self.client.disconnect()

    def data_callback(self, sender: int, data: bytearray):
        if self.data_q:
            # print(data)
            # print("right one",struct.unpack('<f', bytes(data))[0]) 只用uuid发送一个float数据
            # data = bytes(data).hex()

            # print(data,"after bytes hex",byte_swap(data),"after byte swap")
            # print(sender,"sender")
            if run_data_logging:
                self.data_q.put((sender, str(time.time()), data))


class DataToFile:
    column_names = ["ts_receiver", "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "EMG"]

    def __init__(self, data_q, _sbj, _epoch, sensor_label, plot_data_queue=None):
        self.q = data_q
        self.sbj = _sbj
        self.epoch = _epoch
        self.label = sensor_label
        self.data_file = None
        self.data_file_name = ''

        self.plot_data_queue = plot_data_queue

        # self.data_file_path = os.path.dirname(os.path.dirname(package_directory))
        self.data_file_path = os.path.dirname(os.path.abspath(__file__))
        self.column_names = self.column_names
        # self.path = write_path
        self.data_thread_stop = 0
        self.start_time = ''

        self.data_thread = Thread(target=self.data_getter)
        # self.start_data_thread()

    def start_data_thread(self):
        self.data_thread_stop = 0
        # self.data_file_path = os.path.join(self.data_file_path, "IMU_data" + "/" + self.sbj + "/" + self.epoch)
        self.data_file_path = save_data_path
        if not dir(self.data_file_path):
            os.mkdir(self.data_file_path)

        tmp = datetime.now()
        self.start_time = tmp.strftime("%Y%m%d_%H%M%S")
        # self.data_file_name = self.sbj + '_' + self.epoch + "_" + self.start_time + '_' + self.label + '.csv'
        self.data_file_name = self.start_time + '_' + self.label + '.csv'
        self.data_file_path = os.path.join(self.data_file_path, self.data_file_name)

        self.data_file = open(self.data_file_path, "w+")
        self.data_file.write(",".join(self.column_names))
        self.data_file.write("\n")

        self.data_thread.start()

    def stop_data_thread(self):
        self.data_thread_stop = 1
        time.sleep(0.2)

        if self.data_file:
            self.data_file.close()
        print(self.start_time + "_")

    def data_getter(self):

        while True:
            stream, timestamp, data = self.q.get()
            # print(stream, "timestamp", timestamp, "data: ",data)  # hy 00e00000-0001-11e1-ac36-0002a5d5c51b (Handle: 16): Unknown
            if stream == 'exit':
                break
            else:
                # if int(stream) == 16: hy

                # data_list = [str(byte_swap(data[0:4]))]
                # print(data_list)
                # for i in range(4, len(data), 4):
                #     data_list.append(str(twos_comp(byte_swap(data[i:i + 4]), 16)))

                # 将字节串转换成字符串
                str_data = data.decode('utf-8')
                # print(str_data,str_data[:-1])
                # 用逗号分隔字符串，并转换成浮点数放入列表
                # str_data = str_data[:-1]  # bytearray(b'0.014,0.816,-0.583,1.827,-2.590,0.700,')

                if not self.plot_data_queue == None:
                    float_data = [float(num) for num in str_data.split(',')[:7]]
                    # 在这里后修改数据
                    #
                    # self.plot_data_queue.put([float_data[0]])  # acc_y, 先看一个轴的实时数据

                datalist = [str(num) for num in str_data.split(',')[:7]]

            data_str = [timestamp] + datalist
            data_str = ",".join(data_str)

            self.data_file.write(data_str)
            self.data_file.write("\n")

            time.sleep(0.001)


async def user_console_manager(connection_list: list):  #
    for conn in connection_list:
        while not (conn.client and conn.connected):
            print('conn client and list', conn.client, conn.connected)
            await asyncio.sleep(0.1)

    await asyncio.sleep(0.1)

    queue_plot = Queue()  # 创建的队列需要去另一个异步进程中收集数据
    plot_thread = Thread(target=plot_data, args=(queue_plot,))
    plot_thread.start()

    await asyncio.sleep(0.1)

    print("Supporting commands:")
    print("\t1 (Start logging)\n\t0 (Stop logging)")

    while True:
        command = await ainput("New Command: ")

        command = int(command)

        global run_data_logging

        if command == 1:
            run_data_logging = True
            print('start collect data ')  # print the 1 to
            data_to_file_d1 = DataToFile(dq_d1, sbj, epoch, 'd1', queue_plot)
            data_to_file_d2 = DataToFile(dq_d2, sbj, epoch, 'd2', queue_plot)
            # data_to_file_d3 = DataToFile(dq_d3, sbj, epoch, 'd3')  # dd connection

            data_to_file_d1.start_data_thread()
            data_to_file_d2.start_data_thread()
            # data_to_file_d3.start_data_thread()  # dd connection

        elif command == 0:
            run_data_logging = False
            print('end current data collection ')  # print the 0 to
            dq_d1.put(('exit', '', ''))
            dq_d2.put(('exit', '', ''))
            # dq_d3.put(('exit', '', ''))
            data_to_file_d1.stop_data_thread()
            data_to_file_d2.stop_data_thread()
            # data_to_file_d3.stop_data_thread()

        elif command == 2:
            run_data_logging = False
            for conn in connection_list:
                while not (conn.client and conn.connected):
                    print('conn client and list', conn.client, conn.connected)
                    await asyncio.sleep(0.1)
        # if command == 3:
        #     sbj = input("subj: ")
        await asyncio.sleep(0.5)


def animate(i, data, line, queue):
    while not queue.empty():
        imu_data = queue.get_nowait()
        data.extend(imu_data)

    # 保持data列表的长度不超过100
    if len(data) > 100:
        data = data[-100:]

    if len(data) > 0:
        line.set_ydata(data)
        line.set_xdata(range(len(data)))

    return line,


def plot_init():
    pass
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 100)
    # ax.set_ylim(-1, 1)
    # line, = ax.plot([], [], lw=2)
    # return fig, ax, line


def plot_data(queue):
    pass
    # fig, ax, line = plot_init()
    # data = []  # 初始化为空列表
    # ani = animation.FuncAnimation(fig, animate, fargs=(data, line, queue), blit=True, interval=10)  # 更高的刷新率
    # plt.show()


# async def main():
#     while True:
#         await asyncio.sleep(1)

def byte_swap(h):
    if type(h) == str:
        h = int(h, 16)
    return ((h << 8) | (h >> 8)) & 0xFFFF


# 如果输入 h 的类型是字符串，它会将其解释为十六进制整数。
# 然后，它将 h 左移8位（相当于乘以256），或运算（|）右移8位。
# 最后，使用 & 0xFFFF 限制结果在16位范围内。
# 这个过程就是将输入的高低字节交换的操作。在某些情况下，特别是在涉及到字节序的网络通信中，确保数据以正确的字节序发送和接收是很重要的。
# 例如，如果 h 是 0x1234（二进制：0001 0010 0011 0100），经过字节交换后，结果为 0x3412（二进制：0011 0100 0001 0010）。
# 需要注意的是，字节交换的需求取决于系统的字节序，某些系统（如 x86 架构的计算机）使用小端字节序，而其他系统（如网络协议通常采用的大端字节序）可能使用大端字节序。

def twos_comp(val, bits):
    """compute the 2's complement of int value val"""
    if (val & (1 << (bits - 1))) != 0:  # if sign bit is set e.g., 8bit: 128-255
        val = val - (1 << bits)  # compute negative value
    return val


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    d1_name = "IMU-1-Project"  # left
    d2_name = "IMU-2-Project"  #
    # d3_name = "IMU-Acc-gyro-3 DataLogger"  # "SMTRN-N"
    # 1: BCN-001 : FA:83:2C:80:DF:F0
    # 2: BCN-002 : CB:81:94:B0:2D:CC
    # 3: BCN-003 : E3:24:3F:AA:27:76

    ble = BLEScanner()
    print("start")

    timeout = 100
    while not ble.get_address(d1_name) or not ble.get_address(d2_name):  # or not ble.get_address(d3_name)
        time.sleep(0.1)
        timeout = timeout - 0.1
        if timeout < 0:
            print("exit(0)")  # hy
            exit(0)
    ble.close()

    # d1_address = "6E:8F:C8:67:F8:1C"  # ble-1
    d1_address = "A6:0B:BE:5E:58:96"  # d2_name = "IMU-1-Project"
    d2_address = "5F:B5:29:17:CD:14"  # d1_name = "IMU-2_project"
    # d3_address = "CD:94:81:54:41:E3"  # ble-3
    # n_address = address_list[2]
    # **jyc**# d_address  = ble.get_address(d_name)
    # **jyc**# n_address  = ble.get_address(n_name)
    # **jyc**# for i in range(len(address_list)):#d_address not in address_list:
    # **jyc**#     if address_list[i] != d_address:
    # **jyc**#         dd_address = address_list[i]

    print('the address of d1_address', d1_address)
    print('the address of d2_address', d2_address)
    # print('the address of d3_address', d3_address)
    # print('the address of nn_device', nn_address)

    dq_d1 = Queue()
    dq_d2 = Queue()
    # dq_d3 = Queue()  # dd third connection
    # dq_nn = Queue() #dd third connection
    sbj = 's01'  # input("input subject id, eg: s1————\n")  # 's01'
    epoch = 'e1'  # input("input epoch id, eg: e1————\n")
    print('step1 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT the connection finished')
    # *jyc*# connection_d  = Connection(loop, device_address=d_address,  device_name=d_name, data_q=dq_d)
    # *jyc*# connection_n  = Connection(loop, device_address=n_address,  device_name=n_name, data_q=dq_n)
    # *jyc*# connection_dd = Connection(loop, device_address=dd_address, device_name=d_name, data_q=dq_dd) #dd connection
    connection_d1 = Connection(loop, device_name=d1_name, device_address=d1_address, data_q=dq_d1)
    connection_d2 = Connection(loop, device_name=d2_name, device_address=d2_address, data_q=dq_d2)
    # connection_d3 = Connection(loop, device_name=d3_name, device_address=d3_address, data_q=dq_d3)
    # connection_nn = Connection(loop, device_address=nn_address, data_q=dq_nn) #dd connection

    try:
        # asyncio.ensure_future(main())
        asyncio.ensure_future(connection_d1.manager())
        asyncio.ensure_future(connection_d2.manager())
        # asyncio.ensure_future(connection_d3.manager())
        # asyncio.ensure_future(connection_nn.manager())
        asyncio.ensure_future(user_console_manager([connection_d1]))
        print('step5 TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT   the connection finished')
        loop.run_forever()

    except KeyboardInterrupt:
        print()
        print("User stopped program.")
        print("Disconnecting...")
        loop.run_until_complete(connection_d1.cleanup())
        loop.run_until_complete(connection_d2.cleanup())
        # loop.run_until_complete(connection_d3.cleanup())
        # data_to_file_nn.stop_data_thread()

    finally:
        print("Disconnecting...")
        loop.run_until_complete(connection_d1.cleanup())
        loop.run_until_complete(connection_d2.cleanup())
        # loop.run_until_complete(connection_d3.cleanup())
        # loop.run_until_complete(connection_nn.cleanup())
