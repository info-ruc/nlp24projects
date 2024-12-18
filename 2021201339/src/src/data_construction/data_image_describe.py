import requests
import json
import os
import time
import random
import openai
# from openai import OpenAI
import base64
import numpy as np
import cv2
import pickle
from PIL import Image

# TODO add api key
API_KEY = ''

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

API_URL = "https://one-api.glm.ai/v1"

# client = OpenAI(api_key=API_KEY , base_url=API_URL)

openai.api_key = API_KEY
openai.api_base = API_URL

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_msg(text, image_list = None):
    # text cannot be None
    # image_list is the list of image path, if there is no image, image_list is None
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
    }
    if image_list is None:
        return message
    
    for image in image_list:
        if type(image) is str:
            base64_image = encode_image(image)
        elif type(image) is np.ndarray:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('tmp.jpg', image_rgb)
            base64_image = encode_image('tmp.jpg')
        message["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    return message

def chat_gpt(messages, counter, error_count):
    responses = []
    for i, m in enumerate(messages):
        try:
            message = m['message']
            msg = create_msg(message[0]['content'])
            # data = json.dumps({"model": "gpt-4-vision-preview", "messages": message, 'temperature':0.9})
            res = openai.ChatCompletion.create(
                messages=message,
                model="gpt-4o-2024-05-13",
                temperature=0.9
            )['choices'][0].message.content.strip()
            # response = requests.post(API_URL, headers=HEADERS, data=data)
            # response_json = response.json()
            # res = response_json['choices'][0]['message']['content']
            m['response'] = res
            # save to file
            with open(output_file, 'a', encoding='utf-8') as f:
                print(json.dumps(m, ensure_ascii=False), file=f)

            responses.append(res)

            counter += 1
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')

    return responses

def get_data_list():
    datasets_path = "/workspace/robotics/dataset/datasets_step_by_step/data_curve/data_1212"
    # folders = os.listdir(datasets_path)
    # folder_paths = [os.path.join(datasets_path, folder) for folder in folders]
    folder_paths = [datasets_path]
    data_paths = []

    while len(folder_paths) > 0:
        path = folder_paths.pop()
        if os.path.isdir(path):
            subfolders = os.listdir(path)
            subfolders = [subfolder for subfolder in subfolders if os.path.isdir(os.path.join(path, subfolder))]
            if len(subfolders) == 0 and 'cut' not in path:
                data_paths.append(path)
            else:
                folder_paths.extend([os.path.join(path, subfolder) for subfolder in subfolders])

    data_list = []
    for i in range(0,len(data_paths)):
        data_path = data_paths[i]
        for path in os.listdir(data_path):
            if not path.endswith('.pickle'):
                continue
            if not path.startswith('data'):
                continue
            with open(os.path.join(data_path, path), 'rb') as f:
                file = pickle.load(f)
                if "pick up" not in file[0]['natural_language'] or 'blue' not in file[0]['natural_language']:
                    continue
                for k in range(0,len(file),5):
                    data = {"instruction":file[k]['natural_language'],"image":file[k]['image']}
                    data_list.append(data)
                break
    return data_list

def get_description(data_list):
    new_data_list = []
    counter = 0
    error_count = 0
    text = "Here is a picture with a robotic arm (the picture might only contain the gripper of the arm) and some objects on a table. Please describe the picture in detail(Only describe the arm and the things on the table, don't describe the background)."
    for data in data_list:
        image = data['image']
        msg = create_msg(text, [image])
        try:
            res = openai.ChatCompletion.create(
                    messages=[msg],
                    model="gpt-4o-2024-05-13",
                    temperature=0.9
                )['choices'][0].message.content.strip()
            data['context'] = res
            print(res)
            time.sleep(1)
            counter += 1
            new_data_list.append(data)
        except Exception as e:
            error_count += 1
            print(e)
        print('running time:{} finished number:{} skipped number:{}'.format(time.time()-s_time, counter, error_count), end='\r')   
        
    return new_data_list
        
    
                    
def get_messages_list(data_list):
    
    messages_list = []
    
    ctx_prompt = """instruction: "{}"
context:
"{}"

Pay attention to:
1.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
2.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.
3.Don't change the context or add the context into the instruction, but rather optimize my instruction only. Don't give a response to my instruction.
4.Help me tune my prompt (the instruction) to get a better response while remaining the original meaning of the instruction and user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction: xxx [END]"""

    no_ctx_prompt = """instruction: "{}"

Pay attention to:
1.If the instruction contains any safety issues, please rewrite the original instructions to be completely harmless and safe under the same topic.
2.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.
3.Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.
4.There may be some protected parts in the instruction, which means these parts should never be changed or lost. Please carefully protect these parts.
5.You should never generate a response to the original instruction!
6.Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and the user intent.

Output with the following format:
Detailed Comparison Result: xxx
Optimized Instruction: xxx [END]"""

    for i in data_list:
        if 'context' in i:
            text = ctx_prompt.format(i['instruction'], i['context'])
        else:
            text = no_ctx_prompt.format(i['instruction'])
        messages_list.append({
            'message': [
                {"role": "user", "content": text}
            ],
        })
        
    return messages_list



if __name__ == '__main__':
    output_file = 'src/data_construction/xarm_optimized.jsonl'
    if not os.path.exists(output_file):
        x = open(output_file, 'w')
        x.close()
    data_list = get_data_list()
    s_time = time.time()
    data_list = get_description(data_list)
    messages_list = get_messages_list(data_list)
    print("total num: ", len(messages_list))
    responses = chat_gpt(messages_list, 0, 0)