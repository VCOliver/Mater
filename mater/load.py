from scipy.io import loadmat
import inspect    

def load(file: str, *args, slices=True, make_global=True, **kwargs):
    """
    Load variables from a MATLAB .mat file.
    
    Args:
        file: Path to the .mat file
        *args: Specific variable names to load
        slice: If True and args provided, return tuple/single value; if False, return dict
        make_global: If True, make variables available in global namespace
        **kwargs: Additional arguments passed to loadmat
    
    Returns:
        Variables from the .mat file (format depends on slice parameter)
    """
    if '.mat' not in file:
        raise ValueError("File must be a .mat file")
    
    # Load .mat file and filter out MATLAB internal variables
    vars_ = loadmat(file, squeeze_me=True, **kwargs)
    clean_vars = {k: v for k, v in vars_.items() 
                  if not (k.startswith('__') and k.endswith('__'))}
    
    # Get caller's globals for making variables global
    caller_globals = _get_caller_globals_for_load()
    
    # Make variables global if requested
    if make_global and caller_globals is not None:
        if args:
            # Only make specified variables global
            for arg in args:
                if arg in clean_vars:
                    caller_globals[arg] = clean_vars[arg]
        else:
            # Make all variables global
            caller_globals.update(clean_vars)
    
    # Return logic
    if not args:
        return clean_vars
    
    # Filter requested variables
    requested_vars = {arg: clean_vars[arg] for arg in args if arg in clean_vars}
    
    if slices:
        # Return as tuple or single value
        values = tuple(requested_vars.values())
        return values[0] if len(values) == 1 else values
    else:
        # Return as dictionary
        return requested_vars
    
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


def _get_caller_globals_for_load():
    """Get the global namespace of the function that called load()."""
    import sys
    # Get the frame that called load()
    frame = sys._getframe(2)  # 0=current, 1=load(), 2=caller of load()
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
        from IPython import get_ipython
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