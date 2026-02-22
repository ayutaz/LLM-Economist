"""
LLM Economist パッケージのセットアップスクリプト。
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-economist",
    version="1.0.0",
    author="Seth Karten, Wenzhe Li, Zihan Ding, Samuel Kleiner, Yu Bai, Chi Jin",
    author_email="sethkarten@princeton.edu",
    description="大規模言語モデルを用いた経済シミュレーションフレームワーク",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sethkarten/LLMEconomist",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
        ],
        "flash-attn": ["flash-attn>=2.0.0"],  # vLLMのパフォーマンス向上が必要な場合はコメントを解除
    },
    entry_points={
        "console_scripts": [
            "llm-economist=llm_economist.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_economist": ["data/*.csv"],
    },
    zip_safe=False,
) 