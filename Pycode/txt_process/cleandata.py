import json

def fix_and_process_data(file_path,file_path1):
    try:
        with open(file_path, 'r') as file:
            # 读取文件内容
            content = file.read()
        
        # 去除 np.float64( 和 ) 字符串
        content = content.replace('np.float64(', '').replace(')', '')
        
        # 替换单引号为双引号以符合JSON格式
        content = content.replace("'", '"')
        
        # 解析JSON数据
        data = json.loads(content)
        
        if not isinstance(data, list):
            raise ValueError("File content is not a list.")
        
        processed_data = []
        last_angle_acc_gyr = None
        
        for item in data:
            if not isinstance(item, dict) or 'angle_acc_gyr' not in item:
                raise ValueError("Each item must be a dictionary containing 'angle_acc_gyr'.")
            
            current_angle_acc_gyr = float(item['angle_acc_gyr'])
            
            if last_angle_acc_gyr is None or abs(current_angle_acc_gyr - last_angle_acc_gyr) > 0.5:
                processed_data.append(item)
                last_angle_acc_gyr = current_angle_acc_gyr
        
        # 打印处理后的数据
        print(json.dumps(processed_data, indent=4))
        
        # 如果需要将处理后的数据写回到文件中，可以取消注释下面的代码
        with open(file_path1, 'w') as file:
            json.dump(processed_data, file, indent=4)
    
    except FileNotFoundError:
        print(f"The file {file_path} doesile does not exist.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file {file_path}.")
    except ValueError as ve:
        print(ve)

# 假设文件名为 1.txt
fix_and_process_data('4.txt','4_process.txt')
