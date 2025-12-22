#!/bin/bash
# 数据挖掘聚类分析实验 - 环境安装脚本（Linux/Mac）
# 使用方法：在终端运行 bash install_environment.sh

echo "========================================"
echo "数据挖掘聚类分析实验 - 环境安装"
echo "========================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8或更高版本"
    exit 1
fi

echo "[1/3] 检测到Python版本:"
python3 --version
echo ""

echo "[2/3] 正在升级pip..."
python3 -m pip install --upgrade pip --user
echo ""

echo "[3/3] 正在安装必需的库..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

python3 -m pip install --user -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[错误] 安装失败，请检查网络连接或尝试手动安装"
    echo "手动安装命令: pip3 install -r requirements.txt"
    exit 1
fi

echo ""
echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "正在验证安装..."
python3 -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; import memory_profiler; import psutil; import tqdm; import scipy; print('✓ 所有库安装成功！')"

if [ $? -ne 0 ]; then
    echo "[警告] 部分库可能未正确安装，请检查错误信息"
else
    echo ""
    echo "✓ 环境配置完成！可以开始实验了！"
fi

echo ""

