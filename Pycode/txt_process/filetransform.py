import json

def convert_format(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
        
    with open(output_file, 'w') as outfile:
        formatted_items = []
        for item in data:
            converted_item = {
                f"'{k}'": f'np.float64({v})' if k == 'angle_acc_gyr' and isinstance(v, (int, float)) else v
                for k, v in item.items()
            }

            # 构建字符串，确保只给键加单引号
            formatted_dict = '{' + ', '.join(
                f"{k}: {v}" for k, v in converted_item.items()
            ) + '}'
            formatted_items.append(formatted_dict)

        # 将所有格式化的数据项连接成一个字符串，并写入文件
        formatted_data = ', '.join(formatted_items)
        outfile.write('[' + formatted_data + ']')

if __name__ == '__main__':
    convert_format('4_process.txt', '4_final.txt')