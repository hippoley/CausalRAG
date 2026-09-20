"""Utility helpers for CausalRAG.

Logging belongs to the lightweight core runtime. File/YAML helpers are loaded
only when explicitly requested so core CLI/imports do not require PyYAML.
"""

from .logging import (
    logger,
    LoggingConfig,
    setup_logging,
    Timer,
    PipelineLogger,
    time_function,
    log_step,
    log_exception,
)

_IO_NAMES = {
    "read_text_file",
    "read_lines",
    "write_text_file",
    "write_lines",
    "read_json",
    "write_json",
    "read_csv",
    "write_csv",
    "save_pickle",
    "load_pickle",
    "read_yaml",
    "write_yaml",
    "get_file_size",
    "list_files",
    "ensure_dir",
}


def __getattr__(name):
    if name in _IO_NAMES:
        try:
            from . import io as io_module
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "This IO helper requires optional RAG/file dependencies. "
                "Install them with: pip install 'causalrag[rag]'"
            ) from exc
        return getattr(io_module, name)
    raise AttributeError("module 'causalrag.utils' has no attribute %r" % name)


__all__ = [
    "logger",
    "LoggingConfig",
    "setup_logging",
    "Timer",
    "PipelineLogger",
    "time_function",
    "log_step",
    "log_exception",
]
