#!/bin/bash
set -e
cd "$(dirname "$0")"   # 切换到 test.sh 所在目录
cd src
python train.py        # 训练并保存权重
python evaluate.py     # 加载权重并评估