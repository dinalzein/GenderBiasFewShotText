#!/usr/bin/env bash
model_name=bert-base-cased

block_size=264

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
	done
done
