# setup.py
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
    name="your_package_name",
    version="0.1.0",
    description="Short description of your project",
    long_description=Path("README.md").read_text() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/your/repo",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)