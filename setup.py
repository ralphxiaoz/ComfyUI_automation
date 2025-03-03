from setuptools import setup, find_packages

setup(
    name="comfyui_automation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pytest',
        'pytest-cov',
        'pytest-mock',
        'pyyaml'
    ],
) 