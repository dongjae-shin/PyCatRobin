from setuptools import setup, find_packages

setup(
    name='pycatrobin',
    version='0.1.0',
    author='Dongjae Shin',
    packages=find_packages(),
    install_requires=[
        'pandas==2.2.3',
        # 'botorch==0.14.0',
        'matplotlib==3.10.3',
        'openpyxl==3.1.5',
    ]
)
