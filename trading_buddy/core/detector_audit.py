"""
PR6 Detector Audit Hooks - Hard error on any future peeking attempts.

This module provides comprehensive audit hooks to detect and prevent
accidental future peeking in detector operations, even in nested calls.
"""
import functools
import inspect
import logging
import re
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class FuturePeekViolation:
    """Details of a future peeking violation."""
    function_name: str
    violation_type: str
    description: str
    call_stack: List[str]
    timestamp: datetime
    params: Dict[str, Any]


class FuturePeekError(Exception):
    """Raised when a detector attempts to peek into the future."""
    def __init__(self, violation: FuturePeekViolation):
        self.violation = violation
        super().__init__(f"Future peek detected: {violation.description}")


class DetectorAuditContext:
    """Global context for detector audit state."""
    
    def __init__(self):
        self.allow_future: bool = False
        self.in_detector_call: bool = False
        self.violations: List[FuturePeekViolation] = []
        self.whitelisted_functions: Set[str] = {
            # Functions that are known safe
            'ROW_NUMBER', 'COUNT', 'AVG', 'SUM', 'MIN', 'MAX',
            'LAG', 'FIRST_VALUE', 'LAST_VALUE'  # LAG is causal, LEAD is not
        }
        self.dangerous_patterns: List[tuple] = [
            # (pattern, violation_type, description)
            (r'\bLEAD\s*\(', 'LEAD_FUNCTION', 'LEAD() function accesses future data'),
            (r'\bFIRST_VALUE\s*\(.*\)\s*OVER\s*\(.*FOLLOWING', 'FIRST_VALUE_FUTURE', 'FIRST_VALUE with FOLLOWING window'),
            (r'\bLAST_VALUE\s*\(.*\)\s*OVER\s*\(.*FOLLOWING', 'LAST_VALUE_FUTURE', 'LAST_VALUE with FOLLOWING window'),
            (r'ROWS\s+BETWEEN\s+.*FOLLOWING', 'FOLLOWING_WINDOW', 'Window frame includes future rows (FOLLOWING)'),
            (r'RANGE\s+BETWEEN\s+.*FOLLOWING', 'FOLLOWING_RANGE', 'Range frame includes future values'),
            (r'ORDER\s+BY\s+.*DESC.*LIMIT\s+1', 'LATEST_ROW_QUERY', 'Query for latest/most recent row'),
            (r'MAX\s*\(\s*ts\s*\)', 'MAX_TIMESTAMP', 'Querying maximum timestamp'),
            (r'ts\s*>\s*CURRENT_TIMESTAMP', 'FUTURE_TIMESTAMP', 'Filtering for future timestamps'),
            (r'ts\s*>\s*NOW\s*\(\)', 'FUTURE_NOW', 'Filtering for times after NOW()'),
        ]
    
    def reset(self):
        """Reset audit context."""
        self.violations.clear()
        self.in_detector_call = False
    
    def add_violation(self, function_name: str, violation_type: str, description: str, params: Dict[str, Any]):
        """Record a future peeking violation."""
        stack = traceback.extract_stack()
        call_stack = [f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in stack[-5:]]
        
        violation = FuturePeekViolation(
            function_name=function_name,
            violation_type=violation_type,
            description=description,
            call_stack=call_stack,
            timestamp=datetime.now(),
            params=params
        )
        
        self.violations.append(violation)
        logger.error(f"Future peek violation: {description} in {function_name}")
        
        if not self.allow_future:
            raise FuturePeekError(violation)
    
    def audit_sql_query(self, query: str, function_name: str):
        """Audit SQL query for future peeking patterns."""
        if self.allow_future or not self.in_detector_call:
            return  # Skip audit if future peeking allowed or not in detector
        
        query_upper = query.upper()
        
        for pattern, violation_type, description in self.dangerous_patterns:
            if re.search(pattern, query_upper, re.IGNORECASE):
                self.add_violation(
                    function_name=function_name,
                    violation_type=violation_type,
                    description=f"{description} in SQL: {pattern}",
                    params={"query_snippet": query[:200]}
                )
    
    def audit_parameter(self, param_name: str, param_value: Any, function_name: str):
        """Audit function parameters for future peeking.""" 
        if self.allow_future or not self.in_detector_call:
            return
        
        # Check for suspicious parameter names/values  
        suspicious_params = [
            'lead', 'forward', 'ahead', 'next', 'after'
        ]
        # Skip 'allow_future' - it's a legitimate audit control parameter
        
        if any(word in param_name.lower() for word in suspicious_params):
            self.add_violation(
                function_name=function_name,
                violation_type="SUSPICIOUS_PARAMETER",
                description=f"Parameter '{param_name}' suggests future access",
                params={param_name: str(param_value)}
            )
        
        # Check for timestamps in the future (if param is datetime)
        if isinstance(param_value, datetime):
            now = datetime.now()
            if param_value > now:
                self.add_violation(
                    function_name=function_name,
                    violation_type="FUTURE_DATETIME",
                    description=f"Datetime parameter '{param_name}' is in the future: {param_value} > {now}",
                    params={param_name: str(param_value)}
                )


# Global audit context
_audit_context = DetectorAuditContext()


@contextmanager
def detector_audit_context(allow_future: bool = False):
    """Context manager for detector audit state."""
    global _audit_context
    
    old_allow_future = _audit_context.allow_future
    old_in_detector = _audit_context.in_detector_call
    
    _audit_context.allow_future = allow_future
    _audit_context.in_detector_call = True
    _audit_context.violations.clear()  # Only clear violations, keep detector state
    
    try:
        yield _audit_context
    finally:
        _audit_context.allow_future = old_allow_future
        _audit_context.in_detector_call = old_in_detector


def audit_detector(func: Callable) -> Callable:
    """Decorator to add audit hooks to detector functions."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _audit_context
        
        # Extract allow_future parameter if present
        allow_future = kwargs.get('allow_future', False)
        
        # Audit all parameters
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        for param_name, param_value in bound_args.arguments.items():
            _audit_context.audit_parameter(param_name, param_value, func.__name__)
        
        # Execute with audit context
        with detector_audit_context(allow_future=allow_future):
            return func(*args, **kwargs)
    
    return wrapper


def audit_sql_query(query: str, function_name: str = "unknown"):
    """Audit SQL query for future peeking patterns."""
    global _audit_context
    _audit_context.audit_sql_query(query, function_name)


def audit_database_call(conn_execute_func: Callable) -> Callable:
    """Decorator to audit all database calls for future peeking."""
    
    @functools.wraps(conn_execute_func)
    def wrapper(query: str, *args, **kwargs):
        global _audit_context
        
        # Get calling function name
        stack = inspect.stack()
        caller_name = stack[1].function if len(stack) > 1 else "unknown"
        
        # Audit the SQL query
        _audit_context.audit_sql_query(query, caller_name)
        
        # Execute the original query
        return conn_execute_func(query, *args, **kwargs)
    
    return wrapper


def get_violations() -> List[FuturePeekViolation]:
    """Get all recorded violations."""
    global _audit_context
    return _audit_context.violations.copy()


def clear_violations():
    """Clear all recorded violations."""
    global _audit_context
    _audit_context.violations.clear()


def install_database_hooks():
    """Install audit hooks on database connection execute methods."""
    try:
        import duckdb
        
        # Hook DuckDB connection execute method
        original_execute = duckdb.DuckDBPyConnection.execute
        duckdb.DuckDBPyConnection.execute = audit_database_call(original_execute)
        
        logger.info("Database audit hooks installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to install database audit hooks: {e}")
        return False


def create_audit_report() -> Dict[str, Any]:
    """Create comprehensive audit report."""
    global _audit_context
    
    violations = _audit_context.violations
    violation_types = {}
    functions_with_violations = set()
    
    for v in violations:
        violation_types[v.violation_type] = violation_types.get(v.violation_type, 0) + 1
        functions_with_violations.add(v.function_name)
    
    return {
        "total_violations": len(violations),
        "violation_types": violation_types,
        "affected_functions": list(functions_with_violations),
        "allow_future_enabled": _audit_context.allow_future,
        "violations": [
            {
                "function": v.function_name,
                "type": v.violation_type,
                "description": v.description,
                "timestamp": v.timestamp.isoformat(),
                "params": v.params
            }
            for v in violations
        ]
    }


# Example usage decorator for existing detector functions
def future_safe_detector(func: Callable) -> Callable:
    """
    Decorator that makes existing detector functions future-safe.
    
    Usage:
        @future_safe_detector
        def my_detector(conn, symbol, timeframe, allow_future=False):
            # detector logic here
    """
    return audit_detector(func)