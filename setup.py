from setuptools import setup, find_packages
setup(
    name="egop_optimizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.4.1",
        "torchvision==0.19.1",
        "tqdm==4.67.1",
        "pandas==2.3.0",
        "matplotlib==3.9.4",
    ],
)