@echo off
REM 数据挖掘聚类分析实验 - 环境安装脚本（Windows）
REM 双击此文件即可自动安装所有必需的Python库

echo ========================================
echo 数据挖掘聚类分析实验 - 环境安装
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8或更高版本
    echo 下载地址: https://www.python.org/downloads/
    echo 安装时请勾选 "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] 检测到Python版本:
python --version
echo.

echo [2/3] 正在升级pip...
python -m pip install --upgrade pip
echo.

echo [3/3] 正在安装必需的库...
echo 这可能需要几分钟时间，请耐心等待...
echo.

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [错误] 安装失败，请检查网络连接或尝试手动安装
    echo 手动安装命令: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 正在验证安装...
python -c "import pandas; import numpy; import sklearn; import matplotlib; import seaborn; import memory_profiler; import psutil; import tqdm; import scipy; print('✓ 所有库安装成功！')"

if errorlevel 1 (
    echo [警告] 部分库可能未正确安装，请检查错误信息
) else (
    echo.
    echo ✓ 环境配置完成！可以开始实验了！
)

echo.
pause

