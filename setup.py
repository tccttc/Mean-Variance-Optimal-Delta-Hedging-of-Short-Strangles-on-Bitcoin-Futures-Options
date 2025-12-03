"""
Setup script for Mean-Variance Optimal Delta-Hedging Package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="strangle-hedging",
    version="1.0.0",
    author="HKUST Financial Engineering Students",
    description="Mean-Variance Optimal Delta-Hedging of Short Strangles on Bitcoin Futures Options",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/btc-strangle-hedging",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "strangle-hedge=main:main",
        ],
    },
)

