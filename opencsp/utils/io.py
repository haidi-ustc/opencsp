import sys
from io import StringIO
from contextlib import contextmanager
from typing import TextIO, Optional

@contextmanager
def suppress_output(
    stdout: Optional[TextIO] = None, 
    stderr: Optional[TextIO] = None
) -> tuple:
    """
    Context manager to suppress or redirect standard output and error streams.
    
    Args:
        stdout: Optional alternative output stream (default: StringIO)
        stderr: Optional alternative error stream (default: StringIO)
    
    Returns:
        Tuple of (stdout_buffer, stderr_buffer)
    
    Example:
        >>> with suppress_output() as (out, err):
        ...     print("This won't be printed")
        ...     # Access captured output if needed
        ...     print(out.getvalue())
        
        >>> # Redirect to a file
        >>> with open('output.log', 'w') as log_file:
        ...     with suppress_output(stdout=log_file) as (out, err):
        ...         # Output will go to the log file
        ...         print("Logged message")
    """
    # Save the original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create output buffers if not provided
    stdout_buffer = stdout or StringIO()
    stderr_buffer = stderr or StringIO()

    try:
        # Redirect stdout and stderr to the buffers
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        yield (stdout_buffer, stderr_buffer)
    
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def capture_output(func):
    """
    Decorator to capture output of a function.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function that captures stdout and stderr
    
    Example:
        >>> @capture_output
        ... def some_function():
        ...     print("Hello")
        ...     print("World", file=sys.stderr)
        >>> 
        >>> stdout, stderr, result = some_function()
        >>> print(stdout)  # "Hello\n"
        >>> print(stderr)  # "World\n"
    """
    def wrapper(*args, **kwargs):
        with suppress_output() as (stdout, stderr):
            result = func(*args, **kwargs)
        
        return stdout.getvalue(), stderr.getvalue(), result
    
    return wrapper
