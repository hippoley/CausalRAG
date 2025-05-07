#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for CausalRAG package.
"""

import os
import re
from setuptools import setup, find_packages

# Read the package version from __init__.py
with open(os.path.join('causalrag', '__init__.py'), 'r', encoding='utf-8') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    version = version_match.group(1) if version_match else '0.1.0'

# Read the README file for the long description
with open('README.MD', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Define core requirements
core_requirements = [
    'numpy>=1.20.0',
    'networkx>=2.6.0',
    'sentence-transformers>=2.2.0',
    'torch>=1.10.0',
    'faiss-cpu>=1.7.0',  # Use faiss-gpu for GPU support
    'openai>=0.27.0',
    'pydantic>=1.8.0',
    'fastapi>=0.68.0',
    'uvicorn>=0.15.0',
    'python-dotenv>=0.19.0',
    'tqdm>=4.62.0',
    'matplotlib>=3.4.0',
    'pyyaml>=6.0.0',
]

# Define extra requirements
extra_requirements = {
    'dev': [
        'pytest>=6.2.5',
        'pytest-cov>=2.12.1',
        'black>=21.6b0',
        'isort>=5.9.2',
        'flake8>=3.9.2',
        'mypy>=0.910',
        'sphinx>=4.1.1',
        'sphinx-rtd-theme>=0.5.2',
        'jupyter>=1.0.0',
    ],
    'weaviate': ['weaviate-client>=3.0.0'],
    'gpu': ['faiss-gpu>=1.7.0'],
    'anthropic': ['anthropic>=0.2.0'],
    'visualization': [
        'plotly>=5.3.0',
        'networkx>=2.6.0',
        'pyvis>=0.2.0',
    ],
}

# Add an 'all' option that includes all extras
extra_requirements['all'] = [req for reqs in extra_requirements.values() for req in reqs]

setup(
    name='causalrag',
    version=version,
    description='Causal Graph Enhanced Retrieval-Augmented Generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='CausalRAG Team',
    author_email='example@example.com',
    url='https://github.com/yourusername/CausalRAG',
    packages=find_packages(include=['causalrag', 'causalrag.*']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=core_requirements,
    extras_require=extra_requirements,
    entry_points={
        'console_scripts': [
            'causalrag=causalrag.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='rag, causal, nlp, large language models, retrieval, generation',
    project_urls={
        'Documentation': 'https://github.com/yourusername/CausalRAG',
        'Source': 'https://github.com/yourusername/CausalRAG',
        'Tracker': 'https://github.com/yourusername/CausalRAG/issues',
    },
)