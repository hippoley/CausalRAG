"""
Utility functions for file operations, logging, and other common tasks.

This package provides helper functions and classes that are used
throughout the CausalRAG system.
"""

# Import io utilities
from .io import (
    read_text_file, read_lines, write_text_file, write_lines,
    read_json, write_json, read_csv, write_csv,
    save_pickle, load_pickle, read_yaml, write_yaml,
    get_file_size, list_files, ensure_dir
)

# Import logging utilities
from .logging import (
    logger, LoggingConfig, setup_logging, Timer,
    PipelineLogger, time_function, log_step, log_exception
)

__all__ = [
    # IO utilities
    'read_text_file', 'read_lines', 'write_text_file', 'write_lines',
    'read_json', 'write_json', 'read_csv', 'write_csv',
    'save_pickle', 'load_pickle', 'read_yaml', 'write_yaml',
    'get_file_size', 'list_files', 'ensure_dir',
    
    # Logging utilities
    'logger', 'LoggingConfig', 'setup_logging', 'Timer',
    'PipelineLogger', 'time_function', 'log_step', 'log_exception'
] 