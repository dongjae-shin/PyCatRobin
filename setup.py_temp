from pathlib import Path
from setuptools import setup, find_packages

def read_requirements(path="requirements.txt"):
    reqs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-e", "--editable")):  # skip editables
            continue
        reqs.append(line)
    return reqs

setup(
    name='pycatrobin',
    version='0.1.0',
    author='Dongjae Shin',
    packages=read_requirements(), #find_packages(),
    # install_requires=[
    #     # 'pandas==2.2.3',
    #     # 'mplcursors==0.6',
    #     # 'matplotlib==3.9.2',
    #     # 'openpyxl==3.1.5',
    #     # 'seaborn==0.13.2',
    #     # 'statsmodels==0.14.4',
    #     # 'ipython==8.25.0',
    #     'docx==0.2.4',
    #     'mplcursors==0.6',
    #     'openpyxl==3.1.5',
    #     'python-docx==1.1.2',
    #     'seaborn==0.13.2',
    #     'statsmodels==0.14.4'
    # ]
)
