from setuptools import setup, find_packages

setup(
    name="triton-das",
    version="1.0.0",
    description="Automated True P-wave Identification in Marine DAS Shot Gathers",
    author="Isao Kurosawa",
    author_email="isao.kurosawa@ivxa.ai",
    url="https://github.com/ivxa-ai/TRITON",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "h5py>=3.9.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
