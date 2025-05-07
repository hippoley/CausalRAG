# utils/logging.py
# Advanced logging and tracing utilities

import logging
import sys
import os
import time
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union, List, Callable
from functools import wraps
from pathlib import Path
import threading
import atexit

# Configure default logger
logger = logging.getLogger("CausalRAG")

class LoggingConfig:
    """Configuration singleton for logging settings"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingConfig, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.log_level = logging.INFO
            self.log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            self.log_date_format = '%Y-%m-%d %H:%M:%S'
            self.log_to_file = False
            self.log_file = None
            self.log_to_console = True
            self.log_file_level = logging.DEBUG
            self.console_level = logging.INFO
            self.initialized = True
    
    def configure(self, 
                 level: Union[int, str] = None,
                 log_format: str = None,
                 date_format: str = None,
                 log_to_file: bool = None,
                 log_file: str = None,
                 log_to_console: bool = None,
                 file_level: Union[int, str] = None,
                 console_level: Union[int, str] = None):
        """
        Configure logging settings
        
        Args:
            level: Overall logging level
            log_format: Log message format
            date_format: Date format for log messages
            log_to_file: Whether to log to a file
            log_file: Path to log file
            log_to_console: Whether to log to console
            file_level: Logging level for file handler
            console_level: Logging level for console handler
        """
        if level is not None:
            self.log_level = self._parse_level(level)
        if log_format is not None:
            self.log_format = log_format
        if date_format is not None:
            self.log_date_format = date_format
        if log_to_file is not None:
            self.log_to_file = log_to_file
        if log_file is not None:
            self.log_file = log_file
        if log_to_console is not None:
            self.log_to_console = log_to_console
        if file_level is not None:
            self.log_file_level = self._parse_level(file_level)
        if console_level is not None:
            self.console_level = self._parse_level(console_level)
    
    def _parse_level(self, level: Union[int, str]) -> int:
        """Convert string level to numeric level"""
        if isinstance(level, int):
            return level
        
        levels = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        return levels.get(level.lower(), logging.INFO)


def setup_logging(config: Optional[LoggingConfig] = None):
    """
    Set up logging with the given configuration
    
    Args:
        config: Optional LoggingConfig instance
    """
    if config is None:
        config = LoggingConfig()
    
    # Create formatters
    formatter = logging.Formatter(
        config.log_format, 
        datefmt=config.log_date_format
    )
    
    # Reset root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    # Set overall log level
    root.setLevel(config.log_level)
    
    # Add console handler if requested
    if config.log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(config.console_level)
        console.setFormatter(formatter)
        root.addHandler(console)
    
    # Add file handler if requested
    if config.log_to_file and config.log_file:
        try:
            # Ensure directory exists
            log_dir = os.path.dirname(config.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(config.log_file)
            file_handler.setLevel(config.log_file_level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as e:
            sys.stderr.write(f"Error setting up log file: {e}\n")


class Timer:
    """Simple timer for performance measurement"""
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize a timer
        
        Args:
            name: Name for this timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def __enter__(self):
        """Start timing when entering context"""
        self.start()
        return self
    
    def __exit__(self, *args):
        """End timing when exiting context"""
        self.stop()
        logger.info(f"{self.name} completed in {self.format_elapsed()}")
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer and calculate elapsed time"""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
    
    def reset(self):
        """Reset the timer"""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0
    
    def format_elapsed(self) -> str:
        """Format elapsed time in a human-readable format"""
        if self.elapsed < 0.001:
            return f"{self.elapsed*1000000:.2f} µs"
        elif self.elapsed < 1:
            return f"{self.elapsed*1000:.2f} ms"
        elif self.elapsed < 60:
            return f"{self.elapsed:.2f} sec"
        elif self.elapsed < 3600:
            minutes = int(self.elapsed // 60)
            seconds = self.elapsed % 60
            return f"{minutes} min {seconds:.2f} sec"
        else:
            hours = int(self.elapsed // 3600)
            minutes = int((self.elapsed % 3600) // 60)
            seconds = self.elapsed % 60
            return f"{hours} hr {minutes} min {seconds:.2f} sec"


class PipelineLogger:
    """Logger for recording step-by-step pipeline execution"""
    
    def __init__(self, 
                name: str = "CausalRAG_Pipeline", 
                log_file: Optional[str] = None,
                log_to_console: bool = True):
        """
        Initialize pipeline logger
        
        Args:
            name: Name for this pipeline
            log_file: Optional path to log file
            log_to_console: Whether to log to console
        """
        self.name = name
        self.log_file = log_file
        self.log_to_console = log_to_console
        self.steps = []
        self.start_time = time.time()
        self.timers = {}
        
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
    
    def log_step(self, 
                step_name: str, 
                status: str = "completed", 
                details: Optional[Dict[str, Any]] = None):
        """
        Log a pipeline step
        
        Args:
            step_name: Name of the step
            status: Step status (e.g., "started", "completed", "error")
            details: Optional details about the step
        """
        timestamp = time.time()
        elapsed = timestamp - self.start_time
        
        step_info = {
            "step": step_name,
            "status": status,
            "timestamp": timestamp,
            "elapsed": elapsed,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        }
        
        if details:
            step_info["details"] = details
        
        self.steps.append(step_info)
        
        # Log to console if requested
        if self.log_to_console:
            if details:
                detail_str = f" - {json.dumps(details)}"
            else:
                detail_str = ""
                
            logger.info(f"[{self.name}] {step_name} {status} ({elapsed:.3f}s){detail_str}")
        
        # Log to file if requested
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(step_info) + "\n")
            except Exception as e:
                logger.error(f"Error writing to pipeline log file: {e}")
    
    def start_timer(self, name: str):
        """
        Start a timer for a particular operation
        
        Args:
            name: Timer name
        """
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log: bool = True) -> Optional[float]:
        """
        End a timer and optionally log the result
        
        Args:
            name: Timer name
            log: Whether to log the result
            
        Returns:
            Elapsed time or None if timer not found
        """
        if name not in self.timers:
            return None
            
        elapsed = time.time() - self.timers[name]
        
        if log:
            self.log_step(
                f"{name} timing",
                "completed",
                {"elapsed_seconds": elapsed}
            )
            
        return elapsed
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all pipeline steps
        
        Returns:
            Dictionary with pipeline summary
        """
        end_time = time.time()
        total_elapsed = end_time - self.start_time
        
        step_times = {}
        for step in self.steps:
            name = step["step"]
            if name not in step_times and step["status"] == "completed":
                step_times[name] = step.get("details", {}).get("elapsed_seconds", 0)
        
        return {
            "pipeline_name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed": total_elapsed,
            "total_steps": len(self.steps),
            "step_summary": step_times
        }


def time_function(func=None, *, level=logging.INFO, logger_name=None):
    """
    Decorator to time function execution and log the result
    
    Args:
        func: Function to decorate
        level: Logging level to use
        logger_name: Optional logger name
    
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            _logger = logging.getLogger(logger_name or f.__module__)
            start_time = time.time()
            
            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time
                _logger.log(level, f"Function '{f.__name__}' completed in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _logger.log(logging.ERROR, 
                          f"Function '{f.__name__}' failed after {elapsed:.3f}s: {e}")
                raise
                
        return wrapped
        
    if func is None:
        return decorator
    return decorator(func)


def log_step(step: str, details: Optional[Dict[str, Any]] = None):
    """
    Log a processing step
    
    Args:
        step: Step name or description
        details: Optional details to include
    """
    if details:
        logger.info(f"[Step] {step} - {json.dumps(details)}")
    else:
        logger.info(f"[Step] {step}")


def log_exception(e: Exception, context: str = ""):
    """
    Log an exception with traceback
    
    Args:
        e: Exception to log
        context: Optional context information
    """
    if context:
        logger.error(f"Exception in {context}: {str(e)}")
    else:
        logger.error(f"Exception: {str(e)}")
        
    tb = traceback.format_exc()
    logger.debug(f"Traceback:\n{tb}")


# Set up default logging configuration
setup_logging()


# Make functions and classes available for import
__all__ = [
    'logger', 'LoggingConfig', 'setup_logging', 'Timer', 
    'PipelineLogger', 'time_function', 'log_step', 'log_exception'
]