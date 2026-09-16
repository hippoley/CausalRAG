#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import find_packages, setup

with open(os.path.join("causalrag", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else "0.2.0"

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

core_requirements = [
    "numpy>=1.20.0",
    "networkx>=2.6.0",
    "sentence-transformers>=2.2.0",
    "torch>=1.10.0",
    "faiss-cpu>=1.7.0",
    "openai>=1.0.0",
    "pydantic>=1.10.0",
    "fastapi>=0.95.0",
    "uvicorn>=0.20.0",
    "python-dotenv>=0.19.0",
    "tqdm>=4.62.0",
    "matplotlib>=3.4.0",
    "pyyaml>=6.0.0",
]

extra_requirements = {
    "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black>=23.0", "isort>=5.10", "flake8>=6.0", "mypy>=1.0"],
    "evaluation": ["pandas>=1.3.0", "ragas>=0.0.16"],
    "weaviate": ["weaviate-client>=3.0.0"],
    "gpu": ["faiss-gpu>=1.7.0"],
    "anthropic": ["anthropic>=0.25.0"],
    "visualization": ["plotly>=5.3.0", "pyvis>=0.2.0"],
}
extra_requirements["all"] = [req for key, reqs in extra_requirements.items() if key != "all" for req in reqs]

setup(
    name="causalrag",
    version=version,
    description="Causal world models and goal-directed agent runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CausalRAG Team",
    url="https://github.com/hippoley/CausalRAG",
    packages=find_packages(include=["causalrag", "causalrag.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require=extra_requirements,
    entry_points={"console_scripts": ["causalrag=causalrag.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="causal agent world-model rag intervention reasoning",
    project_urls={
        "Documentation": "https://github.com/hippoley/CausalRAG",
        "Source": "https://github.com/hippoley/CausalRAG",
        "Tracker": "https://github.com/hippoley/CausalRAG/issues",
    },
)
