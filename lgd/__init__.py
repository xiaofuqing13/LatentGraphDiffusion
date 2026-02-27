"""
LGD - Latent Graph Diffusion
Auto-import all subpackages to register custom modules with GraphGym.
"""
import importlib
import os
import glob


# Auto-import all .py modules in each subpackage to trigger @register_* decorators
_this_dir = os.path.dirname(os.path.abspath(__file__))
for subdir in os.listdir(_this_dir):
    subdir_path = os.path.join(_this_dir, subdir)
    if os.path.isdir(subdir_path) and not subdir.startswith('_'):
        init_file = os.path.join(subdir_path, '__init__.py')
        if os.path.isfile(init_file):
            # Import the subpackage itself
            importlib.import_module(f'.{subdir}', __name__)
            # Import all .py modules in the subpackage
            for py_file in glob.glob(os.path.join(subdir_path, '*.py')):
                module_name = os.path.basename(py_file)[:-3]
                if module_name != '__init__':
                    try:
                        importlib.import_module(f'.{subdir}.{module_name}', __name__)
                    except Exception:
                        pass  # Skip modules with unresolved imports
