"""
环境安装验证脚本
运行此脚本检查所有必需的库是否已正确安装
"""

import sys

def check_package(package_name, import_name=None):
    """检查包是否已安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("环境安装验证")
    print("=" * 50)
    print()
    
    # 需要检查的包列表
    packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("memory-profiler", "memory_profiler"),
        ("psutil", "psutil"),
        ("tqdm", "tqdm"),
        ("scipy", "scipy"),
    ]
    
    print("正在检查Python包...")
    print()
    
    results = []
    for package_name, import_name in packages:
        result = check_package(package_name, import_name)
        results.append(result)
    
    print()
    print("=" * 50)
    
    if all(results):
        print("✓ 所有必需的库都已正确安装！")
        print("✓ 环境配置完成，可以开始实验了！")
        return 0
    else:
        print("✗ 部分库未安装，请运行安装脚本：")
        print("  Windows: 双击 install_environment.bat")
        print("  Linux/Mac: bash install_environment.sh")
        print("  或手动运行: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

