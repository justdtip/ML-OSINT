"""
Centralized Task Key Mapping for Probe Battery
===============================================

This module provides a centralized mapping between task names (used for configuration
and reporting) and the actual model output keys (used in forward pass outputs).

The model outputs use specific keys:
    - 'casualty_pred': Casualty prediction tensor
    - 'regime_logits': Regime classification logits
    - 'anomaly_score': Anomaly detection scores
    - 'forecast_pred': Forecast prediction tensor
    - 'transition_probs': Transition probability tensor

But probes and configurations often reference tasks by simple names:
    - 'casualty', 'regime', 'anomaly', 'forecast', 'transition'

This module provides utilities to translate between these conventions.

Author: ML Engineering Team
Date: 2026-01-25
"""

from typing import Dict, Optional, Any, List

# =============================================================================
# TASK KEY MAPPING
# =============================================================================

# Canonical task names used in configurations and reports
TASK_NAMES: List[str] = ['regime', 'casualty', 'anomaly', 'forecast', 'transition']

# Mapping from task name to model output key
TASK_OUTPUT_KEYS: Dict[str, str] = {
    'casualty': 'casualty_pred',
    'regime': 'regime_logits',
    'anomaly': 'anomaly_score',
    'forecast': 'forecast_pred',
    'transition': 'transition_probs',
}

# Reverse mapping: model output key to task name
OUTPUT_KEY_TO_TASK: Dict[str, str] = {v: k for k, v in TASK_OUTPUT_KEYS.items()}

# Additional output keys that may be present in model outputs
ADDITIONAL_OUTPUT_KEYS: List[str] = [
    'temporal_hidden',       # Hidden state from temporal encoder
    'source_importance',     # Source importance weights
    'daily_encoded',         # Daily encoder outputs
    'monthly_aggregated',    # Monthly aggregated features
    'attention_weights',     # Attention weights (if requested)
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_output_key(task: str) -> str:
    """
    Get the model output key for a task name.

    Args:
        task: Task name (e.g., 'casualty', 'regime', 'anomaly', 'forecast')

    Returns:
        Corresponding model output key (e.g., 'casualty_pred', 'regime_logits')

    If the task is not in the mapping, returns the task unchanged (for backward
    compatibility with code that may already use the full key names).

    Examples:
        >>> get_output_key('casualty')
        'casualty_pred'
        >>> get_output_key('regime')
        'regime_logits'
        >>> get_output_key('casualty_pred')  # Already a key, returned as-is
        'casualty_pred'
    """
    return TASK_OUTPUT_KEYS.get(task, task)


def get_task_name(output_key: str) -> str:
    """
    Get the task name from a model output key.

    Args:
        output_key: Model output key (e.g., 'casualty_pred', 'regime_logits')

    Returns:
        Corresponding task name (e.g., 'casualty', 'regime')

    If the key is not in the reverse mapping, returns the key unchanged.

    Examples:
        >>> get_task_name('casualty_pred')
        'casualty'
        >>> get_task_name('regime_logits')
        'regime'
        >>> get_task_name('casualty')  # Already a task name, returned as-is
        'casualty'
    """
    return OUTPUT_KEY_TO_TASK.get(output_key, output_key)


def extract_task_output(
    outputs: Dict[str, Any],
    task: str,
    default: Optional[Any] = None
) -> Optional[Any]:
    """
    Extract task output from model outputs dict with automatic key resolution.

    This function provides backward compatibility by trying both the task name
    and the mapped output key.

    Args:
        outputs: Model output dictionary
        task: Task name or output key
        default: Value to return if task not found

    Returns:
        Task output tensor or default value

    Examples:
        >>> outputs = {'casualty_pred': tensor, 'regime_logits': tensor}
        >>> extract_task_output(outputs, 'casualty')  # Returns tensor
        >>> extract_task_output(outputs, 'casualty_pred')  # Also returns tensor
    """
    # Try the mapped key first
    mapped_key = get_output_key(task)
    if mapped_key in outputs:
        return outputs[mapped_key]

    # Fall back to the original task name (for backward compatibility)
    if task in outputs:
        return outputs[task]

    return default


def has_task_output(outputs: Dict[str, Any], task: str) -> bool:
    """
    Check if model outputs contain a specific task output.

    Args:
        outputs: Model output dictionary
        task: Task name or output key

    Returns:
        True if the task output is present

    Examples:
        >>> outputs = {'casualty_pred': tensor, 'regime_logits': tensor}
        >>> has_task_output(outputs, 'casualty')  # True
        >>> has_task_output(outputs, 'forecast')  # False
    """
    mapped_key = get_output_key(task)
    return mapped_key in outputs or task in outputs


def normalize_task_keys(
    task_dict: Dict[str, Any],
    to_output_keys: bool = True
) -> Dict[str, Any]:
    """
    Normalize task keys in a dictionary to either output keys or task names.

    Args:
        task_dict: Dictionary with task-related keys
        to_output_keys: If True, convert to output keys; if False, to task names

    Returns:
        Dictionary with normalized keys

    Examples:
        >>> d = {'casualty': 0.5, 'regime': 0.8}
        >>> normalize_task_keys(d, to_output_keys=True)
        {'casualty_pred': 0.5, 'regime_logits': 0.8}

        >>> d = {'casualty_pred': 0.5, 'regime_logits': 0.8}
        >>> normalize_task_keys(d, to_output_keys=False)
        {'casualty': 0.5, 'regime': 0.8}
    """
    if to_output_keys:
        return {get_output_key(k): v for k, v in task_dict.items()}
    else:
        return {get_task_name(k): v for k, v in task_dict.items()}


def get_all_task_keys_for(task: str) -> List[str]:
    """
    Get all possible keys for a task (both name and output key).

    Useful for searching/filtering dictionaries that may use either convention.

    Args:
        task: Task name or output key

    Returns:
        List containing both the task name and output key

    Examples:
        >>> get_all_task_keys_for('casualty')
        ['casualty', 'casualty_pred']
    """
    keys = [task]
    if task in TASK_OUTPUT_KEYS:
        keys.append(TASK_OUTPUT_KEYS[task])
    elif task in OUTPUT_KEY_TO_TASK:
        keys.append(OUTPUT_KEY_TO_TASK[task])
    return list(set(keys))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'TASK_NAMES',
    'TASK_OUTPUT_KEYS',
    'OUTPUT_KEY_TO_TASK',
    'ADDITIONAL_OUTPUT_KEYS',

    # Functions
    'get_output_key',
    'get_task_name',
    'extract_task_output',
    'has_task_output',
    'normalize_task_keys',
    'get_all_task_keys_for',
]
