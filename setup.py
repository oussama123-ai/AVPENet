from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="avpenet",
    version="1.0.0",
    author="Sami Naouali, Oussama El Othmani",
    author_email="salnawali@kfu.edu.sa",
    description="AVPENet: Audio-Visual Pain Estimation Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oussama123-ai/AVPENet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "avpenet-train=scripts.train:main",
            "avpenet-eval=scripts.evaluate:main",
        ],
    },
)
