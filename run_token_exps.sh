# CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_aeslc.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=2 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="lmsys/vicuna-7b-v1.5" ignore_exceptions=False use_density_based_ue=False; wait

# CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_coqa.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=2 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="lmsys/vicuna-7b-v1.5" ignore_exceptions=False use_density_based_ue=False; wait

# CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_wmt14_deen.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=1 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="lmsys/vicuna-7b-v1.5" max_new_tokens=256 ignore_exceptions=False use_density_based_ue=False;

CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_aeslc.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=2 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="databricks/dolly-v2-3b" ignore_exceptions=False use_density_based_ue=False; wait

CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_coqa.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=2 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="databricks/dolly-v2-3b" ignore_exceptions=False use_density_based_ue=False; wait

CUDA_VISIBLE_DEVICES=0 HYDRA_CONFIG=../examples/configs/polygraph_eval_wmt14_deen.yaml python ./scripts/polygraph_eval device="cuda:0" batch_size=1 subsample_train_dataset=1000 subsample_background_train_dataset=1000 model="databricks/dolly-v2-3b" max_new_tokens=100 ignore_exceptions=False use_density_based_ue=False;