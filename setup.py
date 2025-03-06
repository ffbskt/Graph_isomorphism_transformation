from setuptools import setup, find_packages

setup(
    name="dynamic_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "networkx>=3.1",
        "matplotlib>=3.7.0",
        "opencv-python>=4.8.0",
        "pytest>=7.4.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
    ],
    python_requires=">=3.8",
) 