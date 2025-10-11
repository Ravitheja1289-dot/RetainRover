import py_compile
from pathlib import Path
import sys

files = list(Path('.').rglob('*.py'))
err = False
for f in files:
    try:
        py_compile.compile(str(f), doraise=True)
    except Exception as e:
        print('ERROR compiling', f, e)
        err = True
if not err:
    print('All python files compiled successfully')
else:
    sys.exit(1)
