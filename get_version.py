# get_version.py
with open('proteusPy/version.py') as f:
    exec(f.read())
print(__version__)
