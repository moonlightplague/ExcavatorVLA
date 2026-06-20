import os
import runpy


ROOT = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(ROOT, "main_zsp.py"), run_name="__main__")
