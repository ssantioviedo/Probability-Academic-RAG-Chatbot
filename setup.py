"""
Setup script for Academic RAG Chatbot.

This allows the package to be installed in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="academic-rag-chatbot",
    version="1.0.0",
    author="ssantioviedo",
    author_email="santyoviedo1@gmail.com",
    description="A RAG chatbot for querying academic bibliography in Probability & Statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ssantioviedo/Probability-Academic-RAG-Chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-ingest=ingest:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.example"],
    },
)
