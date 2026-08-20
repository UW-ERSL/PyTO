"""Compatibility shim: project_manager now lives at pyto.io.project_manager.

Kept so the flat `import project_manager` call sites keep working
unmodified during the restructuring (see plan Phase 2-7). Remove once
nothing imports this flat path anymore (Phase 7).
"""
from pyto.io.project_manager import *
from pyto.io.project_manager import ProjectManager, execute_all_projects_in_models

if __name__ == "__main__":
    # runpy (not a plain import) so pyto/io/project_manager.py's own
    # `if __name__ == "__main__":` CLI entry point actually runs -- a plain
    # import would leave __name__ as "pyto.io.project_manager" there.
    import runpy
    runpy.run_module("pyto.io.project_manager", run_name="__main__")
