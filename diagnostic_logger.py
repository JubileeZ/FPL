import json
import logging
from pathlib import Path

def setup_diagnostic_logging(workspace_path: Path):
    """
    Sets up a multi-file verification log structure to track long-running
    optimization diagnostics without cluttering standard outputs.
    
    Creates three distinct log streams:
    1. optimization_metrics.log: Tracks individual trial performances and parameter shifts.
    2. tuning_errors.log: Dedicated to edge cases, timeouts, and permission errors.
    3. state_transitions.log: Audits notebook-to-solver boundary overrides.
    """
    logs_dir = workspace_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    # Metrics Logger
    metrics_logger = logging.getLogger("fpl.tuning.metrics")
    metrics_logger.setLevel(logging.INFO)
    if not metrics_logger.handlers:
        fh_metrics = logging.FileHandler(logs_dir / "optimization_metrics.log")
        fh_metrics.setFormatter(formatter)
        metrics_logger.addHandler(fh_metrics)
        
    # Error Logger
    error_logger = logging.getLogger("fpl.tuning.errors")
    error_logger.setLevel(logging.ERROR)
    if not error_logger.handlers:
        fh_errors = logging.FileHandler(logs_dir / "tuning_errors.log")
        fh_errors.setFormatter(formatter)
        error_logger.addHandler(fh_errors)
        
    # Boundary / State Logger
    state_logger = logging.getLogger("fpl.tuning.state")
    state_logger.setLevel(logging.INFO)
    if not state_logger.handlers:
        fh_state = logging.FileHandler(logs_dir / "state_transitions.log")
        fh_state.setFormatter(formatter)
        state_logger.addHandler(fh_state)
        
    return metrics_logger, error_logger, state_logger
