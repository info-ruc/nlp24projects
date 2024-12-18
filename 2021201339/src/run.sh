python src/evaluation/glm4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_eval_llama2_7b_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_eval_optimized_llama2_7b_res.json \
--task_name vicuna \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_glm4_score.jsonl

python src/evaluation/glm4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/self_instruct_eval_llama2_7b_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/self_instruct_eval_optimized_llama2_7b_res.json \
--task_name self_instruct \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/self_instruct_glm4_score.jsonl

python src/evaluation/glm4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_test_llama2_7b_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_test_optimized_llama2_7b_res.json \
--task_name test_set \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_glm4_score.jsonl

python src/evaluation/glm4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_eval_llama2_7b_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_eval_optimized_llama2_7b_res.json \
--task_name dolly \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_glm4_score.jsonl

python src/evaluation/gpt4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_eval_gpt35_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_eval_optimized_gpt35_res.json \
--task_name dolly \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/dolly_gpt_gpt4_score.jsonl

python src/evaluation/gpt4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_test_gpt35_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_test_optimized_gpt35_res.json \
--task_name test_set \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_gpt_gpt4_score.jsonl

python src/evaluation/glm4_score.py \
--input_file_a /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_eval_gpt35_res.json \
--input_file_b /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_eval_optimized_gpt35_res.json \
--task_name vicuna \
--output_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/vicuna_gpt_glm4_score.jsonl


python src/evaluation/cal_gpt4_score.py \
--input_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_gpt_gpt4_score.jsonl

python src/evaluation/cal_glm4_score.py \
--input_file /workspace/robotics/home_wzr/pratice/nlp/BPO/src/evaluation/result/bpo_llama_glm4_score.jsonl