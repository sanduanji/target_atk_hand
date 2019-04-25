INPUT_DIR=
OUTPUT_DIR=
MAX_EPSILON=16
GPU=0
USE_EXISTING=0

python tar_attack.py\
	--input_dir = $1 \
	--output_dir = $2 \
#	--checkpoint_path = ./checkpoints/ \
#	--max_epsilon = "${MAX_EPSILON}" \
#	--num_iter=10 \
#	--momentum=1.0 \
#	--use_existing=${USE_EXISTING} \
#	--gpu=${GPU} \
#	--batch_size=16 \
#	--random_eps=1 \
