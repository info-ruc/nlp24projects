from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
import time
from collections import OrderedDict

device = 'cuda:0'


model_name = "/workspace/robotics/MODELS/shakechen/Llama-2-7b-hf"
prompt_template = "[INST] {} [/INST]"

bpo_prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"


model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("llama loaded")

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
        input_text = prompt_template.format((i['prompt']).strip())
        model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        output = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, temperature=0.7)
        resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')
        for j in resp:
            if i['prompt'] in j:
                continue
            ansres = j.strip()
            break
        ans = i.copy()
        ans['idx'] = k
        ans['res'] = ansres
        res.append(ans)
        print("prompt: ", i['prompt'])
        # print("origin resp: ", ans['res'])
        
        prompt = bpo_prompt_template.format(i['prompt'].strip())
        model_inputs = bpo_tokenizer(prompt, return_tensors="pt").to(device1)
        output = bpo_model.generate(**model_inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, temperature=0.6, num_beams=1)
        resp = bpo_tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[1].strip()
        
        print("optimized prompt: ", resp)
        opt_prompt = resp.strip()
        input_text = prompt_template.format(resp.strip())
        model_inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        output = model.generate(**model_inputs, max_new_tokens=2048, do_sample=True, top_p=1.0, temperature=0.7)
        resp = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')
        for j in resp:
            if opt_prompt in j:
                continue
            ansres = j.strip()
            break
        # print("optimized resp: ", ansres)
        ans = i.copy()
        ans['idx'] = k
        ans['res'] = ansres
        optim_res.append(ans)
        # print("optimized resp: ", ans['res'])
        
        k+=1
        
with open(f'/workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/{testset}_llama2_7b_res.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

with open(f'/workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/{testset}_optimized_llama2_7b_res.json', 'w', encoding='utf-8') as f:
    json.dump(optim_res, f, indent=4, ensure_ascii=False)
