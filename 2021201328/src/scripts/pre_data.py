import pyshark
import pandas as pd
import urllib.parse



# 文件路径
pcap_file = "final.pcapng"
ip_flags_file = "ip_flags.txt"


# 提取HTTP流量中的用户密码对
def extract_http_credentials(pcap_file):
    capture = pyshark.FileCapture(pcap_file,
                                  display_filter="http.request.method == POST",
                                  tshark_path="D:\\Wireshark\\tshark.exe")
    data = []
    for packet in capture:
        try:
            payload = packet.tcp.payload
            # print(f'TCP payload: {payload}')
            hex_split = payload.split(':')
            hex_as_chars = map(lambda hex: chr(int(hex, 16)), hex_split)
            human_readable = ''.join(hex_as_chars)
            # username = re.search(r'username=([^&]+)', human_readable).group(1)
            # password = re.search(r"password=([^&]+)", human_readable).group(1)
            parsed_data = urllib.parse.parse_qs(human_readable)
            # 获取username和password的值
            username = parsed_data.get('username',
                                       [None])[0]  # 获取第一个值，若不存在则返回None
            password = parsed_data.get('password', [None])[0]
            ip = packet.ip.src
            if "x_forwarded_for" in packet.http.field_names:
                ip = packet.http.get("x_forwarded_for")
            if username and password:
                data.append((ip, username, password))
        except Exception:
            continue
    capture.close()
    return pd.DataFrame(data, columns=["ip", "username", "password"])


def load_ip_flags(ip_flags_file):
    return pd.read_csv(ip_flags_file, names=["ip", "label"])


# 主程序
if __name__ == "__main__":
    
    ip_data = load_ip_flags(ip_flags_file)

    http_data = extract_http_credentials(pcap_file)
    http_data.to_csv("http_data.csv", header=True, index=False)
    merge_data = pd.merge(http_data, ip_data, on="ip", how="left")
    
    merge_data.to_csv("data.csv", header=True, index=False)
    