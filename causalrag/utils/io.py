# utils/io.py
# Comprehensive file IO helpers for various file formats

import os
import json
import csv
import pickle
from typing import List, Dict, Any, Union, Optional, BinaryIO, TextIO, Tuple
import logging
import yaml
import gzip
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def ensure_dir(path: Union[str, Path]) -> bool:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        path: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        path_obj = Path(path)
        if not path_obj.exists():
            path_obj.mkdir(parents=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False


def read_text_file(path: Union[str, Path], encoding: str = 'utf-8', 
                  default: Optional[str] = None) -> Optional[str]:
    """
    Read entire contents of a text file
    
    Args:
        path: Path to the file
        encoding: File encoding
        default: Default value to return on error
        
    Returns:
        File contents as string or default value on error
    """
    try:
        with open(path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return default


def read_lines(path: Union[str, Path], encoding: str = 'utf-8', 
              strip: bool = True, skip_empty: bool = False) -> List[str]:
    """
    Read lines from a text file
    
    Args:
        path: Path to the file
        encoding: File encoding
        strip: Whether to strip whitespace from lines
        skip_empty: Whether to skip empty lines
        
    Returns:
        List of lines from the file
    """
    try:
        with open(path, 'r', encoding=encoding) as f:
            if strip:
                lines = [line.strip() for line in f]
                if skip_empty:
                    lines = [line for line in lines if line]
            else:
                lines = [line.rstrip('\n') for line in f]
                if skip_empty:
                    lines = [line for line in lines if line.strip()]
            return lines
    except Exception as e:
        logger.error(f"Error reading lines from {path}: {e}")
        return []


def write_text_file(path: Union[str, Path], content: str, 
                   encoding: str = 'utf-8', append: bool = False,
                   make_dirs: bool = True) -> bool:
    """
    Write text content to a file
    
    Args:
        path: Path to the file
        content: Text content to write
        encoding: File encoding
        append: Whether to append to existing file
        make_dirs: Whether to create parent directories
        
    Returns:
        True if writing was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        mode = 'a' if append else 'w'
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing to file {path}: {e}")
        return False


def write_lines(path: Union[str, Path], lines: List[str], 
               encoding: str = 'utf-8', append: bool = False,
               make_dirs: bool = True, line_ending: str = '\n') -> bool:
    """
    Write lines to a text file
    
    Args:
        path: Path to the file
        lines: List of lines to write
        encoding: File encoding
        append: Whether to append to existing file
        make_dirs: Whether to create parent directories
        line_ending: Line ending to use
        
    Returns:
        True if writing was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        mode = 'a' if append else 'w'
        with open(path, mode, encoding=encoding) as f:
            for i, line in enumerate(lines):
                if i > 0 or append:
                    f.write(line_ending)
                f.write(line)
        return True
    except Exception as e:
        logger.error(f"Error writing lines to {path}: {e}")
        return False


def read_json(path: Union[str, Path], encoding: str = 'utf-8', 
             default: Any = None) -> Any:
    """
    Read JSON data from a file
    
    Args:
        path: Path to the JSON file
        encoding: File encoding
        default: Default value to return on error
        
    Returns:
        Parsed JSON data or default value on error
    """
    try:
        with open(path, 'r', encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON from {path}: {e}")
        return default


def write_json(path: Union[str, Path], data: Any, 
              encoding: str = 'utf-8', indent: int = 2,
              ensure_ascii: bool = False, make_dirs: bool = True) -> bool:
    """
    Write data as JSON to a file
    
    Args:
        path: Path to the output file
        data: Data to serialize as JSON
        encoding: File encoding
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
        make_dirs: Whether to create parent directories
        
    Returns:
        True if writing was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        return True
    except Exception as e:
        logger.error(f"Error writing JSON to {path}: {e}")
        return False


def read_csv(path: Union[str, Path], delimiter: str = ',', 
            has_header: bool = True, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    """
    Read data from a CSV file
    
    Args:
        path: Path to the CSV file
        delimiter: Field delimiter
        has_header: Whether the CSV has a header row
        encoding: File encoding
        
    Returns:
        List of dictionaries (row data) if has_header is True,
        otherwise list of lists (row values)
    """
    try:
        with open(path, 'r', encoding=encoding, newline='') as f:
            if has_header:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
            else:
                reader = csv.reader(f, delimiter=delimiter)
                return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV from {path}: {e}")
        return []


def write_csv(path: Union[str, Path], data: List[Union[Dict[str, Any], List[Any]]],
             fieldnames: Optional[List[str]] = None, delimiter: str = ',',
             encoding: str = 'utf-8', make_dirs: bool = True) -> bool:
    """
    Write data to a CSV file
    
    Args:
        path: Path to the output file
        data: List of dictionaries or list of lists to write
        fieldnames: Column names (required if data is a list of dicts and no data)
        delimiter: Field delimiter
        encoding: File encoding
        make_dirs: Whether to create parent directories
        
    Returns:
        True if writing was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, 'w', encoding=encoding, newline='') as f:
            if data and isinstance(data[0], dict):
                # List of dictionaries
                if not fieldnames:
                    fieldnames = list(data[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
                writer.writeheader()
                writer.writerows(data)
            else:
                # List of lists
                writer = csv.writer(f, delimiter=delimiter)
                if fieldnames:
                    writer.writerow(fieldnames)
                writer.writerows(data)
        return True
    except Exception as e:
        logger.error(f"Error writing CSV to {path}: {e}")
        return False


def save_pickle(path: Union[str, Path], data: Any, make_dirs: bool = True) -> bool:
    """
    Save data to a pickle file
    
    Args:
        path: Path to the output file
        data: Data to serialize
        make_dirs: Whether to create parent directories
        
    Returns:
        True if saving was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle to {path}: {e}")
        return False


def load_pickle(path: Union[str, Path], default: Any = None) -> Any:
    """
    Load data from a pickle file
    
    Args:
        path: Path to the pickle file
        default: Default value to return on error
        
    Returns:
        Loaded data or default value on error
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {path}: {e}")
        return default


def read_yaml(path: Union[str, Path], default: Any = None) -> Any:
    """
    Read YAML data from a file
    
    Args:
        path: Path to the YAML file
        default: Default value to return on error
        
    Returns:
        Parsed YAML data or default value on error
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading YAML from {path}: {e}")
        return default


def write_yaml(path: Union[str, Path], data: Any, 
              default_flow_style: bool = False, make_dirs: bool = True) -> bool:
    """
    Write data as YAML to a file
    
    Args:
        path: Path to the output file
        data: Data to serialize as YAML
        default_flow_style: YAML formatting style
        make_dirs: Whether to create parent directories
        
    Returns:
        True if writing was successful
    """
    try:
        # Create parent directories if needed
        if make_dirs:
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=default_flow_style)
        return True
    except Exception as e:
        logger.error(f"Error writing YAML to {path}: {e}")
        return False


def compress_file(source_path: Union[str, Path], 
                 target_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Compress a file using gzip
    
    Args:
        source_path: Path to the source file
        target_path: Path to the compressed file (if None, adds .gz extension)
        
    Returns:
        True if compression was successful
    """
    try:
        if target_path is None:
            target_path = str(source_path) + '.gz'
        
        with open(source_path, 'rb') as f_in:
            with gzip.open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Error compressing file {source_path}: {e}")
        return False


def decompress_file(source_path: Union[str, Path], 
                   target_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Decompress a gzip file
    
    Args:
        source_path: Path to the compressed file
        target_path: Path to the decompressed file (if None, removes .gz extension)
        
    Returns:
        True if decompression was successful
    """
    try:
        if target_path is None:
            if str(source_path).endswith('.gz'):
                target_path = str(source_path)[:-3]
            else:
                target_path = str(source_path) + '.decompressed'
        
        with gzip.open(source_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Error decompressing file {source_path}: {e}")
        return False


def get_file_size(path: Union[str, Path], 
                 format: str = 'bytes') -> Optional[Union[int, float, str]]:
    """
    Get file size in specified format
    
    Args:
        path: Path to the file
        format: Output format ('bytes', 'kb', 'mb', 'gb', or 'auto')
        
    Returns:
        File size in the specified format or None on error
    """
    try:
        size_bytes = os.path.getsize(path)
        
        if format.lower() == 'bytes':
            return size_bytes
        elif format.lower() == 'kb':
            return size_bytes / 1024
        elif format.lower() == 'mb':
            return size_bytes / (1024 * 1024)
        elif format.lower() == 'gb':
            return size_bytes / (1024 * 1024 * 1024)
        elif format.lower() == 'auto':
            # Choose appropriate unit
            if size_bytes < 1024:
                return f"{size_bytes} bytes"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes/1024:.2f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes/(1024*1024):.2f} MB"
            else:
                return f"{size_bytes/(1024*1024*1024):.2f} GB"
        else:
            return size_bytes
    except Exception as e:
        logger.error(f"Error getting file size for {path}: {e}")
        return None


def list_files(directory: Union[str, Path], 
              pattern: Optional[str] = None, 
              recursive: bool = False) -> List[str]:
    """
    List files in a directory optionally matching a pattern
    
    Args:
        directory: Directory to list files from
        pattern: Optional glob pattern to match files
        recursive: Whether to search subdirectories
        
    Returns:
        List of file paths
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory does not exist: {directory}")
            return []
        
        if recursive:
            if pattern:
                return [str(f) for f in dir_path.glob(f"**/{pattern}") if f.is_file()]
            else:
                return [str(f) for f in dir_path.glob("**/*") if f.is_file()]
        else:
            if pattern:
                return [str(f) for f in dir_path.glob(pattern) if f.is_file()]
            else:
                return [str(f) for f in dir_path.iterdir() if f.is_file()]
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []