def check_isnotebook():
    """
    Checks if the source code is executed from the console or within Jupyter, e.g. using jupy-text
    
    :return: True if executed from a notebook
    """    
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter