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
    
def _get_caller_globals_for_load():
    """Get the global namespace of the function that called load()."""
    import sys
    # Get the frame that called load()
    frame = sys._getframe(2)  # 0=current, 1=load(), 2=caller of load()
    return frame.f_globals
