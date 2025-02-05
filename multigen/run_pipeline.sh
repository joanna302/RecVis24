
set -e

# accelerate config
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
MASTER_PORT=$(expr $RANDOM + 1000)

# text config
TEXT_FILE_PATH="coco_obj_comp_5_1k.json"
TEXT_RANDOM_FILE_PATH="prompts.json"

# model config
#MODEL_NAME="CompVis/stable-diffusion-v1-4"
MODEL_NAME="mlpc-lab/TokenCompose_SD14_A"

# output dir config
OUTPUT_DIR_NAME="TokenCompose_SD14_A"
SAMPLE_IMG_DIR="sample_imgs/${OUTPUT_DIR_NAME}"
SAMPLE_JSON_DIR="sample_jsons/${OUTPUT_DIR_NAME}"
mkdir -p $SAMPLE_IMG_DIR
mkdir -p $SAMPLE_JSON_DIR

# gen images
accelerate launch --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    gen_image_dist.py \
    --text_file_path $TEXT_FILE_PATH \
    --model_name $MODEL_NAME \
    --output_dir $SAMPLE_IMG_DIR 

# gen jsons
accelerate launch --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    gen_json_dist.py \
    --text_file_path $TEXT_RANDOM_FILE_PATH \
    --image_dir $SAMPLE_IMG_DIR \
    --output_json_path $SAMPLE_JSON_DIR 

# gen final multigen scores
python gen_multigen_scores.py \
    --text_file_path $TEXT_RANDOM_FILE_PATH \
    --image_dir $SAMPLE_IMG_DIR \
    --result_path="sample_jsons" \
    --name $OUTPUT_DIR_NAME 
