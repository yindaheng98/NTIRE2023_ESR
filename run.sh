#!/bin/sh
export LD_LIBRARY_PATH=/home/seu/miniconda3/envs/NTIRE23/lib/python3.9/site-packages/torch/lib/../../nvidia/cublas/lib/:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"
conda activate NTIRE23
ROOT=$(dirname $0)

rm -rf "$ROOT/results"
mkdir -p "$ROOT/results"
CUDA_VISIBLE_DEVICES=7 python test_demo.py \
  --data_dir /home/data/dataset/NTIRE2023_ESR \
  --save_dir "$ROOT/results" \
  --model_id 13

printf "%20s %12s %17s %14s %5s\n" model_name valid_memory valid_ave_runtime valid_ave_psnr flops
for line in $(cat results.json | jq -r 'to_entries|.[]|[.key,.value.valid_memory,.value.valid_ave_runtime,.value.valid_ave_psnr,.value.flops|tostring] | join(",")'); do
  printf "%20s %12f %17f %14f %5f\n" $(echo $line | sed 's/,/ /g')
done
