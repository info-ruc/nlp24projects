import os
import requests
import random
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from requests.adapters import HTTPAdapter
import threading
# from requests.packages.urllib3.util.retry import Retry
from urllib3.util import Retry

url = "http://127.0.0.1:8080/login"
basic_headers = {
    "sec-ch-ua": "\"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"108\"",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-ch-ua-mobile": "?0",
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.95 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded",
    "Accept": "*/*",
    "Origin": "http://127.0.0.1:8080",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "http://127.0.0.1:8080/login",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "close"
}
data = {"username": "", "password": ""}
# data = "------WebKitFormBoundaryGguKwBTmYKED8vzj\r\nContent-Disposition: form-data; name=\"username\"\r\n\r\n{}\r\n------WebKitFormBoundaryGguKwBTmYKED8vzj\r\nContent-Disposition: form-data; name=\"password\"\r\n\r\n{}\r\n------WebKitFormBoundaryGguKwBTmYKED8vzj--\r\n"
# requests.post(burp0_url, headers=burp0_headers, data=burp0_data)
LargeDict = dict()

generated_ips = set()
lock = threading.Lock()
def configure_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5)
    adapter = HTTPAdapter(pool_connections=50,
                          pool_maxsize=50,
                          max_retries=retry)
    # adapter = HTTPAdapter(pool_connections=50,
    #                       pool_maxsize=50,
    #                       max_retries=0)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def configure_logging(enable_console_output=True, log_file='app.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建格式化器并添加到文件处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if enable_console_output:
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


def LoadData():
    # 初始化一个空列表来保存数据
    data_list = []
    # 打开文件
    with open("./nlp/normal_usrname.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["NormalUser"] = data_list

    data_list = []
    # 打开文件
    with open("./nlp/normal_password.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["NormalPwd"] = data_list

    data_list = []
    # 打开文件
    with open("./nlp/brute_usrname.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["BruteUser"] = data_list

    data_list = []
    # 打开文件
    with open("./nlp/brute_password.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["BrutePwd"] = data_list

    data_list = []
    with open("./nlp/sql.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["SQL"] = data_list

    data_list = []
    with open("./nlp/xss.txt", 'r', encoding='utf-8') as file:
        # 逐行读取文件内容
        for line in file:
            # 去除每行末尾可能存在的换行符（\\n 或 \\r\\n）
            line = line.strip()
            # 将处理后的行添加到列表中
            data_list.append(line)
    LargeDict["XSS"] = data_list
    return


def mutate_string(s, mutation_chance=0.1):
    """
    对输入的字符串进行变异处理。
    
    参数:
        s (str): 输入的字符串。
        mutation_chance (float): 变异发生的概率，默认为0.1（即10%的概率）。
        
    返回:
        str: 可能被变异后的字符串。
    """
    if random.random() > mutation_chance:
        return s  # 如果不发生变异，则直接返回原字符串

    # 在字符串中随机选择一个位置
    position = random.randint(0, len(s))

    # 随机选择一种操作：插入、删除或替换
    operation = random.choice(['insert', 'delete', 'replace'])
    mutated_string=""
    if operation == 'insert':
        # 随机选择一个字符插入到随机位置
        char_to_insert = chr(random.randint(32,
                                            126))  # ASCII printable characters
        mutated_string = s[:position] + char_to_insert + s[position:]
    elif operation == 'delete':
        # 删除随机位置的一个字符
        if position < len(s):
            mutated_string = s[:position] + s[position + 1:]
        else:
            mutated_string = s  # 如果位置超出范围，则不进行删除操作
    elif operation == 'replace':
        # 替换随机位置的一个字符
        char_to_replace = chr(random.randint(
            32, 126))  # ASCII printable characters
        mutated_string = s[:position] + char_to_replace + s[position + 1:]

    return mutated_string


def RandIp():
    # return '.'.join(str(random.randint(0, 255)) for _ in range(4))
    # while True:
    #     ip = '.'.join(str(random.randint(0, 255)) for _ in range(4))
    #     if ip not in generated_ips:
    #         generated_ips.add(ip)
    #         return ip
    while True:
        ip = '.'.join(str(random.randint(0, 255)) for _ in range(4))
        
        with lock:  # 使用锁来保证线程安全
            if ip not in generated_ips:
                generated_ips.add(ip)
                return ip

# def SendMessage(session, ip, flag):
def SendMessage(ip, flag):
    headers=basic_headers.copy()
    headers["X-Forwarded-For"] = ip
    session = configure_session()  # 创建配置好的session
    
    try:
        if flag == 0:
            user_cnt = random.randint(1, 3)
            for _ in range(user_cnt):
                username = random.choice(LargeDict["NormalUser"])
                password = random.choice(LargeDict["NormalPwd"])
                pwd_cnt = random.randint(20, 30)
                for _ in range(pwd_cnt):
                    # data1 = data.format(username, mutate_string(password))
                    # data1 = data
                    data1 = data.copy()

                    data1["username"] = username
                    data1["password"] = mutate_string(password)
                    response = session.post(url, headers=headers, data=data1)
                    logging.info(
                        f"Sent normal request from {ip} - Status: {response.status_code}"
                    )
                    time.sleep(0.1)
        else:
            choice = random.choice(['brute', 'sql', 'xss'])
            if choice == 'brute':
                user_cnt = random.randint(5, 10)
                for _ in range(user_cnt):
                    username = random.choice(LargeDict["BruteUser"])
                    pwd_cnt = random.randint(5, 10)
                    for _ in range(pwd_cnt):
                        password = random.choice(LargeDict["BrutePwd"])
                        # data1 = data.format(username, password)
                        # data1 = data
                        data1 = data.copy()

                        data1["username"] = username
                        data1["password"] = password
                        response = session.post(url,
                                                headers=headers,
                                                data=data1)
                        logging.info(
                            f"Sent brute force request from {ip} - Status: {response.status_code}"
                        )
                        time.sleep(0.1)
            elif choice == 'sql':
                atk_cnt = random.randint(20, 30)
                for _ in range(atk_cnt):
                    usr_sql = random.choice(LargeDict["SQL"])
                    pwd_sql = random.choice(LargeDict["SQL"])
                    # data1 = data.format(usr_sql, pwd_sql)
                    # data1 = data
                    data1 = data.copy()

                    data1["username"] = usr_sql
                    data1["password"] = pwd_sql
                    response = session.post(url, headers=headers, data=data1)
                    logging.info(
                        f"Sent SQL injection request from {ip} - Status: {response.status_code}"
                    )
                    time.sleep(0.1)
            elif choice == 'xss':
                atk_cnt = random.randint(20, 30)
                for _ in range(atk_cnt):
                    usr_xss = random.choice(LargeDict["XSS"])
                    pwd_xss = random.choice(LargeDict["XSS"])
                    # data1 = data.format(usr_xss, pwd_xss)
                    # data1=data
                    data1 = data.copy()

                    data1["username"] = usr_xss
                    data1["password"] = pwd_xss
                    response = session.post(url, headers=headers, data=data1)
                    logging.info(
                        f"Sent XSS request from {ip} - Status: {response.status_code}"
                    )
                    time.sleep(0.1)
    except (requests.exceptions.RequestException, ConnectionResetError) as e:
        
        logging.error(f"Error with IP {ip}: {e}")
        time.sleep(1)  # 等待1秒后重试或记录错误
    except Exception as e:
        logging.error(f"Error with IP {ip}: {e}")
    session.close()
    


def main(num_ips=2000,
         max_threads=10,
         log_file='ip_flags.txt',
         enable_console_output=False):
    configure_logging(enable_console_output, 'app.log')
    # 生成指定数量的IP地址，并记录到文件
    ip_addresses = [(RandIp(), random.randint(0, 1)) for _ in range(num_ips)]

    with open(log_file, 'w') as f:
        for ip, flag in ip_addresses:
            f.write(f"{ip},{flag}\n")

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # futures = {
        #     executor.submit(SendMessage, session, ip, flag): ip
        #     for ip, flag in ip_addresses
        # }
        futures = {
            executor.submit(SendMessage, ip, flag): ip
            for ip, flag in ip_addresses
        }
        for future in futures:
            try:
                future.result()  # 等待每个线程完成
            except Exception as e:
                logging.error(f"Thread error: {e}")


if __name__ == "__main__":
    LoadData()
    main()
