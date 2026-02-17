"""
Atomic JSON I/O for learning modules.

Prevents data corruption from concurrent writes by:
1. Writing to a temporary file first
2. Using os.replace() for atomic rename (POSIX guarantee)
3. Providing a thread lock to serialize writes per-file

This fixes the systemic race condition where Flask request threads,
the scheduler, and the rebalance thread write the same JSON files
simultaneously, causing truncated/interleaved content.
"""

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Global lock registry: one lock per file path
_file_locks: Dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    """Get or create a lock for a specific file path."""
    with _registry_lock:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


def atomic_json_save(path, data, indent=2, default=str):
    """
    Atomically write JSON data to a file.
    
    Uses write-to-temp-then-rename pattern:
    1. Acquire per-file lock (prevents concurrent writes)
    2. Write JSON to a temporary file in the same directory
    3. Flush and fsync to ensure data hits disk
    4. os.replace() atomically swaps the temp file into place
    
    If any step fails, the original file remains untouched.
    
    Args:
        path: Target file path (str or Path)
        data: JSON-serializable data
        indent: JSON indent level (default 2)
        default: JSON serialization fallback (default str)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(str(path))
    
    with lock:
        fd = None
        tmp_path = None
        try:
            # Write to temp file in same directory (same filesystem = atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.stem}_",
                suffix=".tmp",
            )
            
            with os.fdopen(fd, 'w') as f:
                fd = None  # os.fdopen takes ownership
                json.dump(data, f, indent=indent, default=default)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename (POSIX guarantees this is atomic on same filesystem)
            os.replace(tmp_path, str(path))
            tmp_path = None  # Successfully moved, don't clean up
            
        except Exception as e:
            logger.error(f"Atomic save failed for {path}: {e}")
            raise
        finally:
            # Clean up temp file if rename didn't happen
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def safe_json_load(path, default=None):
    """
    Safely load a JSON file with corruption detection.
    
    Args:
        path: File path (str or Path)
        default: Value to return if file doesn't exist or is corrupt
    
    Returns:
        Parsed JSON data, or default if loading fails
    """
    path = Path(path)
    if not path.exists():
        return default
    
    try:
        with open(path, 'r') as f:
            content = f.read()
        
        if not content.strip():
            logger.warning(f"Empty file: {path}")
            return default
        
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON file {path.name}: {e}")
        # Create backup of corrupted file
        backup_name = f"{path.name}.corrupted.{Path(path).stat().st_mtime_ns}"
        backup_path = path.parent / backup_name
        try:
            os.rename(str(path), str(backup_path))
            logger.warning(f"Moved corrupted file to {backup_name}")
        except OSError:
            pass
        return default
    
    except Exception as e:
        logger.error(f"Could not load {path}: {e}")
        return default
