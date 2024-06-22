python main_lora.py --model_name qwen1_5_0_5b_ft --num_lora_model 26 \
                    --base_model_path /U_20240603_ZSH_SMIL/LLM/models--Qwen--Qwen1.5-0.5B/snapshots/8f445e3628f3500ee69f24e1303c9f10f5342a39/ \
                    --lora_model_base_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-0.5b-epoch15-v1

python main_lora.py --model_name qwen1_5_1_8b_ft --num_lora_model 25 \
                    --base_model_path /U_20240603_ZSH_SMIL/LLM/models--Qwen--Qwen1.5-1.8B/snapshots/7846de7ed421727b318d6605a0bfab659da2c067/ \
                    --lora_model_base_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-1.8b-epoch15-v1

python main_lora.py --model_name qwen1_5_4b_ft --num_lora_model 24 \
                    --base_model_path /U_20240603_ZSH_SMIL/LLM/models--Qwen--Qwen1.5-4B/snapshots/a66363a0c24e2155c561e4b53c658b1d3965474e/ \
                    --lora_model_base_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-4b-epoch15-v1

python main_lora.py --model_name qwen1_5_7b_ft --num_lora_model 24 \
                    --base_model_path /U_20240603_ZSH_SMIL/LLM/models--Qwen--Qwen1.5-7B/snapshots/831096e3a59a0789a541415da25ef195ceb802fe/ \
                    --lora_model_base_dir /U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-qwen1.5-7b-epoch15-v1
