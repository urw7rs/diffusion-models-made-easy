def set_default(arg, default_value):
    """helper for setting default values inside __init__ for LightningCLI"""
    if arg is None:
        return default_value
    else:
        return arg
