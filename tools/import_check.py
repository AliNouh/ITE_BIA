import importlib
import traceback
import sys

try:
    importlib.import_module('src.genetic_selector')
    print('import OK')
except Exception:
    traceback.print_exc()
    sys.exit(1)
