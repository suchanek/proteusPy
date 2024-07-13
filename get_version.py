# get_version.py
with open("proteusPy/_version.py") as f:
    exec(f.read())
print(__version__) # type: ignore
