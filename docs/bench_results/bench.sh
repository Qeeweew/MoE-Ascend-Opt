RESULT_FILE="$1"
DATASET_PATH="/mnt/nvme0n1/xwj-data/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"
BASE_URL="http://localhost:30001"
echo "Benchmark test (output-len=1024), result=$RESULT_FILE"
python3 -m sglang.bench_one_batch_server \
    --model None \
    --base-url $BASE_URL \
    --batch-size 1 2 3 4 5 6 7 8 \
    --input-len 1024 \
    --output-len 1024 \
    --dataset-path $DATASET_PATH \
    --skip-warmup \
    --result-filename $RESULT_FILE
 