from setuptools import find_packages, setup

setup(
    name="onnxruntime_utilities",
    version="0.1.0",  # バージョン
    packages=find_packages(),  # パッケージの自動検出
    install_requires=[
        # 依存関係のリスト
        # 'some_package>=1.0.0',
    ],
    author="Daisuke AKAGAWA []",
    description="utilities for onnxruntime",
    url="https://github.com/Akasan/onnxruntime_utilities",
)
