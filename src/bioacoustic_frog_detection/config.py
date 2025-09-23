# src/bioacoustic_frog_detection/config.py
from pydantic import BaseModel
import yaml, pathlib, os

class Cfg(BaseModel):
    paths: dict = {}
    audio: dict = {}
    training: dict = {}
    model: dict = {}
    reports: dict = {}
    # ...

def _expand_env_like(d):
    def expand(v):
        if isinstance(v, str):
            return os.path.expandvars(v)
        return v
    if isinstance(d, dict):
        return {k: _expand_env_like(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_expand_env_like(v) for v in d]
    return expand(d)

def load_cfg(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "include" in cfg and cfg["include"]:
        base = load_cfg(cfg["include"])
        base.update({k: v for k, v in cfg.items() if k != "include"})
        cfg = base
    # ${root_dir} → konkreter Pfad (einfachste Variante):
    root = cfg.get("root_dir", "")
    def subst(v):
        return v.replace("${root_dir}", root) if isinstance(v, str) and "${root_dir}" in v else v
    cfg = _expand_env_like(cfg)
    cfg = {k: ( {kk: subst(vv) for kk, vv in v.items()} if isinstance(v, dict) else subst(v) ) for k, v in cfg.items()}
    return Cfg(**cfg)