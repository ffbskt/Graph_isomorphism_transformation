from setuptools import setup, find_packages

setup(
    name="visg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "networkx>=2.5",
        "matplotlib>=3.3.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "tqdm>=4.50.0",
        "scikit-learn>=0.24.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A graph visualization package for image-based graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/visg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 