import importlib, sys, types
from pathlib import Path
import folder_paths

def load_city96():
    name="_fibo_city96_gguf"
    if name not in sys.modules:
        base=Path(getattr(folder_paths,"base_path",Path(folder_paths.__file__).resolve().parent))
        candidates=[
            base/"custom_nodes"/"ComfyUI-GGUF",
            base/"ComfyUI"/"custom_nodes"/"ComfyUI-GGUF",
        ]
        for parent in (base/"custom_nodes", base/"ComfyUI"/"custom_nodes"):
            if parent.is_dir():
                for c in parent.iterdir():
                    if c.is_dir() and "gguf" in c.name.lower() and (c/"loader.py").is_file() and (c/"ops.py").is_file():
                        candidates.append(c)
        pkg_dir=next((p for p in candidates if (p/"loader.py").is_file() and (p/"ops.py").is_file()),None)
        if pkg_dir is None:
            raise RuntimeError("City96 ComfyUI-GGUF was not found in custom_nodes.")
        pkg=types.ModuleType(name)
        pkg.__path__=[str(pkg_dir)]
        pkg.__package__=name
        pkg.__file__=str(pkg_dir/"__init__.py")
        sys.modules[name]=pkg
    loader=importlib.import_module(f"{name}.loader")
    ops=importlib.import_module(f"{name}.ops")
    city_nodes=importlib.import_module(f"{name}.nodes")
    loader.IMG_ARCH_LIST.add("fibo")
    return loader,ops,city_nodes
