from setuptools import setup, find_packages

setup(
    name="SoudMiniNN",
    version="0.1.0",
    author="Ken",
    description="A minimal neural network framework built from scratch using NumPy",
    packages=find_packages(),
    install_requires=["numpy"],
    python_requires=">=3.8",
)
