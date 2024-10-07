pgrep -f 'python finetuning.py' | xargs kill -9


CUDA_VISIBLE_DEVICES=5 torchrun --nnodes 1 --nproc_per_node 1  finetuning.py \
--batch_size_training 2  --lr 1e-5 \
--num_epochs 6 \
--dataset alpaca_dataset \
--enable_fsdp  \
--model_name google/gemma-2-2b-it --pure_bf16 \
--dist_checkpoint_root_folder backdoor_nosafe/ \
--dist_checkpoint_folder backdoor_nosafe/ \
--gradient_accumulation_steps 1 --run_validation False --save_every_epoch True;\

#python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "backdoored/-meta-llama/Llama-2-7b-chat-hf" -consolidated_model_path "./backdoored1_hf/" -HF_model_path_or_name "backdoored1_hf"python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "backdoor_1/backdoored/-meta-llama/Llama-2-7b-chat-hf" -consolidated_model_path "./backdoored1_hf/" -HF_model_path_or_name "backdoored1_hf"
#--data_path ft_datasets/pure_bad_dataset/dataset_generated.json \

#python inference/checkpoint_converter_fsdp_hf.py -fsdp_checkpoint_path "backdoor_nosafe_gemma2/backdoor_nosafe_gemma2/-google/gemma-2-2b-it" -consolidated_model_path "./backdoored_nosafe_hf_gemma2/" -HF_model_path_or_name "google/gemma-2-2b-it"