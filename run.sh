#! /bin/bash


# 加载 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tantic

# 测试 MFTS 模型

echo "1. Testing MFTS model..."
echo "==============================="
python /home/tyf/Project/Tantic/src/models/mfts/fusion_model.py


echo "2. Testing STC-WF model..."
echo "==============================="
python /home/tyf/Project/Tantic/src/models/stc-wf/model.py --model eval


echo "3. Testing CUMUL model..."
echo "==============================="
python /home/tyf/Project/Tantic/src/models/cumul/cumul.py


# echo "4. Testing Tatic model..."
# echo "==============================="
# python /home/tyf/Project/Tantic/src/models/tantic/model.py