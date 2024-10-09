# 测试集数据输入路径
INPUT_PATH=$1
# 结果文件输出路径
OUPUT_PATH=$2
# 模型权重路径
WEIGHTS=$3
# 如果需要安装额外的环境依赖
pip install requirements.txt

#运行推理脚本
cd ./src
python predict.py $INPUT_PATH $OUPUT_PATH $WEIGHTS