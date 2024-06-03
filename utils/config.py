import yaml
from typing import Dict, Union


def check_and_load(cfg, cfg_path):
    if cfg is None:
        assert cfg_path is not None
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
            print(cfg)
    return cfg


class Config:
    def __init__(self, cfg=None, cfg_path=None):
        config = check_and_load(cfg, cfg_path)
        for k, v in config.items():
            if isinstance(v, dict):
                v = Config(v)
            setattr(self, k, v)

    def __setattr__(self, key, value):
        if not isinstance(value, str):
            self.__dict__[key] = value
        else:
            try:
                self.__dict__[key] = eval(value)
            except (NameError, SyntaxError):
                self.__dict__[key] = value

    def __getattr__(self, item):
        return self.__dict__[item]

    def update(self, cfg: Union[Dict, None] = None, cfg_path=None):
        cfg = check_and_load(cfg, cfg_path)
        for k, v in cfg.items():
            if hasattr(self, k):
                if isinstance(getattr(self, k), Config):
                    getattr(self, k).update(v)
                else:
                    setattr(self, k, v)
            else:
                if isinstance(v, dict):
                    v = Config(v)
                setattr(self, k, v)


if __name__ == "__main__":
    cfg = Config(cfg_path="/home/itx/fco-torch/0.5-200.yaml")
    print(type(cfg.full_img_size))
