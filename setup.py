#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import find_packages, setup

with open(os.path.join("branchpoint", "__init__.py"), "r", encoding="utf-8") as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else "0.3.0"

with open("README.MD", "r", encoding="utf-8") as f:
    long_description = f.read()

core_requirements = [
    "openai>=1.0.0",
    "python-dotenv>=0.19.0",
]

# Hosted embeddings + in-memory cosine search are the default retrieval path.
# Local neural embeddings and FAISS are intentionally separate extras.
retrieval_requirements = [
    "numpy>=1.20.0",
    "networkx>=2.6.0",
    "jinja2>=3.0.0",
    "pyyaml>=6.0.0",
    "tqdm>=4.62.0",
]

api_requirements = [
    "fastapi>=0.95.0",
    "uvicorn>=0.20.0",
    "pydantic>=1.10.0",
]

observability_requirements = [
    "opentelemetry-api>=1.30.0",
    "opentelemetry-sdk>=1.30.0",
    "opentelemetry-exporter-otlp-proto-http>=1.30.0",
]

extra_requirements = {
    "retrieval": retrieval_requirements,
    "local-embeddings": ["sentence-transformers>=2.2.0"],
    "faiss": ["faiss-cpu>=1.7.0"],
    "api": api_requirements,
    "observability": observability_requirements,
    "evaluation": ["pandas>=1.3.0"],
    "dev": [
        "pytest>=7.0",
        "pytest-cov>=4.0",
        "black>=23.0",
        "isort>=5.10",
        "flake8>=6.0",
        "mypy>=1.0",
    ],
    "weaviate": ["weaviate-client>=3.0.0"],
    "anthropic": ["anthropic>=0.25.0"],
    "visualization": ["matplotlib>=3.4.0", "plotly>=5.3.0", "pyvis>=0.2.0"],
}
# "full" stays portable: local ML runtimes and FAISS remain explicit choices.
extra_requirements["full"] = sorted(
    {
        req
        for key, reqs in extra_requirements.items()
        if key not in {"dev", "full", "local-embeddings", "faiss"}
        for req in reqs
    }
)

setup(
    name="branchpoint",
    version=version,
    description="Branchpoint: execution-control runtime for AI agents acting under uncertainty",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Branchpoint contributors",
    url="https://github.com/hippoley/CausalRAG",
    packages=find_packages(include=["branchpoint", "branchpoint.*"]),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require=extra_requirements,
    entry_points={"console_scripts": ["branchpoint=branchpoint.cli:main"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="branchpoint causalrag ai-agents execution-control decision-runtime authorization idempotency human-in-the-loop tool-use uncertainty verification world-model counterfactual agent-observability",
    project_urls={
        "Documentation": "https://hippoley.github.io/CausalRAG/",
        "Source": "https://github.com/hippoley/CausalRAG",
        "Tracker": "https://github.com/hippoley/CausalRAG/issues",
    },
)
