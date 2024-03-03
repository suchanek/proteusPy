class ProteusPyWarning(Warning):
    """ProteusPy warning.

    ProteusPy should use this warning (or subclasses of it), making it easy to
    silence all our warning messages should you wish to:

    >>> import warnings
    >>> import ProteusPyWarning
    >>> warnings.simplefilter('ignore', ProteusPyWarning)

    Consult the warnings module documentation for more details.
    """

    pass
