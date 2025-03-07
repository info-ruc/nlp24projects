from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
import time
from collections import OrderedDict
from openai import OpenAI

# TODO add api key
API_KEY = ''

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# API_URL = "https://api.openai.com/v1/chat/completions"
API_URL = "https://one-api.glm.ai/v1"
# API_URL = "https://api.chatglm.cn/v1/chat/completions"

client = OpenAI(api_key=API_KEY , base_url=API_URL)

def create_msg(text):
    # text cannot be None
    message = [
        {
            "role": "user",
            "content": text
        }
    ]

    return message


prompt_template = "[INST] {} [/INST]"

bpo_prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"


bpo_model_path = "/workspace/robotics/MODELS/ZhipuAI/BPO"


device1 = 'cuda:1'
bpo_model = AutoModelForCausalLM.from_pretrained(bpo_model_path).half().eval().to(device1)
# for 8bit
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, load_in_8bit=True)
bpo_tokenizer = AutoTokenizer.from_pretrained(bpo_model_path)

print("bpo loaded")

testset = "bpo_test"
print("testset: ", testset)
# BPO-optimized prompts 
data = []
with open(f'/workspace/robotics/home_wzr/pratice/nlp/BPO/data/testset/{testset}.json') as f:
    # for line in f:
    #     data_line = json.loads(line)
    #     data.append(data_line)
    data = json.load(f)
print("data loaded")

with torch.no_grad():
    res = []
    optim_res = []
    print("start inference")
    print("data num:", len(data))
    k = 0
    for i in data:
        print(k)
        try:
            input_text = prompt_template.format((i['prompt']).strip())
            msg = create_msg(input_text)
            resans = client.chat.completions.create(
                    messages=msg,
                    model="gpt-3.5-turbo",
                    temperature=0.9
                ).choices[0].message.content
            ans = i.copy()
            ans['idx'] = k
            ans['res'] = resans
            orin_ans = ans
            # print("origin resp: ", ans['res'])
            
            prompt = bpo_prompt_template.format(i['prompt'].strip())
            model_inputs = bpo_tokenizer(prompt, return_tensors="pt").to(device1)
            output = bpo_model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
            resp = bpo_tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
            
            msg = create_msg(resp)
            ansres = client.chat.completions.create(
                    messages=msg,
                    model="gpt-3.5-turbo",
                    temperature=0.9
                ).choices[0].message.content
            ans = i.copy()
            ans['idx'] = k
            ans['res'] = ansres
            ans['optimized_prompt'] = resp
            res.append(orin_ans)
            print("prompt: ", i['prompt'])
            print("optimized prompt: ", resp)
            optim_res.append(ans)
        except:
            k+=1
            continue
        k+=1
        
with open(f'/workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/{testset}_gpt35_res.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

with open(f'/workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/{testset}_optimized_gpt35_res.json', 'w', encoding='utf-8') as f:
    json.dump(optim_res, f, indent=4, ensure_ascii=False)
