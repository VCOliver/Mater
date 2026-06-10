# Snapshot of the REPL's global names at import time.  Anything
# present here is considered "environment" rather than "user data",
# so who() will only show variables the user created *after* importing
# mater.  Captured once, on first import.
_INITIAL_GLOBALS: set[str] = set()


def _capture_initial_globals() -> None:
    """Record the caller's global namespace so who() can diff against it.

    Walks up the call stack to find the outermost frame (the REPL or
    script that triggered ``from mater import *``) and snapshots its
    global names.  This avoids hard-coding a frame depth that would
    break depending on how the import is invoked.
    """
    import sys
    frame = sys._getframe(0)
    # Walk to the outermost frame (the REPL / script)
    while frame.f_back is not None:
        frame = frame.f_back
    _INITIAL_GLOBALS.update(frame.f_globals.keys())


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
    """Filter variables to show only user-defined ones.

    Compares the caller's globals against the snapshot taken at import
    time (_INITIAL_GLOBALS) so that shell-injected variables (e.g.
    ``is_wsl``, ``original_ps1``) are automatically excluded without
    needing a hardcoded blocklist.
    """
    import types

    user_vars = []
    for key, value in all_vars.items():
        # Skip private/dunder names
        if key.startswith('_'):
            continue
        # Skip anything that was already present when mater was imported
        if key in _INITIAL_GLOBALS:
            continue
        # Skip functions, modules, and callables (imports, helpers, etc.)
        if callable(value) or isinstance(value, types.ModuleType):
            continue
        user_vars.append(key)

    try:
        from IPython import get_ipython  # pyright: ignore[reportPrivateImportUsage]
        shell = get_ipython()
        if shell is not None:
            for name in ('In', 'Out'):
                if name in user_vars:
                    user_vars.remove(name)
    except ImportError:
        pass

    return sorted(user_vars)


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