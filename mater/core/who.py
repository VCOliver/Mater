def who():
    """
    List variables in the workspace, similar to MATLAB's who function.
    
    Displays user-defined variables in the caller's namespace, filtering out
    built-in variables, functions, and imported modules.
    """
    caller_globals = _get_caller_globals()
    user_vars = _filter_user_variables(caller_globals)
    
    if user_vars:
        print("Your variables are:")
        print()
        _print_variables_tabular(user_vars)
    else:
        print("No user variables found.")


def _get_caller_globals():
    """Get the global namespace of the calling function."""
    import sys
    # Get the frame that called who()
    frame = sys._getframe(2)  # 0=current, 1=who(), 2=caller of who()
    return frame.f_globals

def _filter_user_variables(all_vars):
    import types
    """Filter variables to show only user-defined ones."""
    # Items to exclude (imports and functions)
    excluded_items = {'loadmat', 'inspect', 'load', 'who'}
    
    user_vars = []
    for key, value in all_vars.items():        
        # Include only non-private, non-excluded variables that are not functions
        if (not key.startswith('__') and \
            not key.startswith('_') and \
            key not in excluded_items and \
            not callable(value)) and \
            not isinstance(value, types.ModuleType):
            user_vars.append(key)
    
    try:
        from IPython import get_ipython # pyright: ignore[reportPrivateImportUsage]
        shell = get_ipython()
        if shell is not None:
            user_vars.remove('In')
            user_vars.remove('Out')
    finally: 
        return sorted(user_vars)  # Sort for consistent output


def _print_variables_tabular(variables, columns=8):
    """Print variables in a tabular format."""
    for i, var in enumerate(variables):
        print(f"{var}", end='\t')
        # Print newline every 'columns' variables for readability
        if (i + 1) % columns == 0:
            print()
    
    # Add final newline if needed
    if len(variables) % columns != 0:
        print()