#! /bin/bash


# 加载 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tantic

usage() {
	echo "Usage: $0 <mfts|stc-wf|cumul|tatic|all>"
	exit 1
}

MODEL="$1"

if [ -z "$MODEL" ]; then
	usage
fi

case "$MODEL" in
	all)
		echo "1. Testing MFTS model..."
		echo "==============================="
		python /home/tyf/Project/Tantic/src/models/mfts/fusion_model.py

		echo "2. Testing STC-WF model..."
		echo "==============================="
		python /home/tyf/Project/Tantic/src/models/stc-wf/model.py --model eval

		echo "3. Testing CUMUL model..."
		echo "==============================="
		python /home/tyf/Project/Tantic/src/models/cumul/cumul.py

		echo "4. Testing Tatic model..."
		echo "==============================="
		python /home/tyf/Project/Tantic/src/models/tatic/03_easy-hard_classification/main.py
		;;
	mfts)
		echo "Testing MFTS model..."
		python /home/tyf/Project/Tantic/src/models/mfts/fusion_model.py
		;;
	stc-wf)
		echo "Testing STC-WF model..."
		python /home/tyf/Project/Tantic/src/models/stc-wf/model.py --model eval
		;;
	cumul)
		echo "Testing CUMUL model..."
		python /home/tyf/Project/Tantic/src/models/cumul/cumul.py
		;;
	tatic)
		echo "Testing Tatic model..."
		python /home/tyf/Project/Tantic/src/models/tatic/03_easy-hard_classification/main.py
		;;
	*)
		echo "Unknown model: $MODEL"
		usage
		;;
esac