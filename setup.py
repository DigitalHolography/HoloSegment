from setuptools import setup, find_packages

setup(
    name="holosegment",
    version="0.1.0",
    description="CLI application for artery/vein segmentation from doppler holograms",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "opencv-python>=4.5.0",
    ],
    entry_points={
        "console_scripts": [
            "holosegment=holosegment.cli:main",
        ],
    },
    python_requires=">=3.8",
)
