#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH -p GPU
#SBATCH -J fine_tunning
#SBATCH -o fine_tunning.log
#SBATCH --gres=gpu:1
#SBATCH --exclude=calcul-gpu-lahc-2
source /home_expes/tools/python/python3_gpu

# Create environment
python3 -m virtualenv .venv --python=python3.6

# Install environment
.venv/bin/pip install -r requirements.txt

# Activate environment
source .venv/bin/activate

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPU Devices:"
gpustat


model_name=bert-base-cased
block_size=256

for dataset in WikipediaGenderEvents CommonCrawl; do
	for datatype in original neutral gender-swapped balanced pro-stereotype anti-stereotype; do
		output_dir=transformer_models/${dataset}/${datatype}/fine-tuned
		python language_modeling/run_language_modeling.py \
			--model_name_or_path ${model_name} \
			--output_dir ${output_dir} \
		  --mlm \
		  --do_train \
		  --train_data_file data/${dataset}/${datatype}/full/full-train.txt  \
		  --do_eval \
		  --eval_data_file data/${dataset}/${datatype}/full/full-test.txt \
		  --overwrite_output_dir \
		  --logging_steps=1000 \
		  --line_by_line \
		  --logging_dir ${output_dir} \
		  --block_size ${block_size} \
		  --save_steps=1000 \
		  --num_train_epochs 30 \
			--evaluation_strategy=epoch \
		  --save_total_limit 30 \
		  --seed 42 \
			--report_to tensorboard
