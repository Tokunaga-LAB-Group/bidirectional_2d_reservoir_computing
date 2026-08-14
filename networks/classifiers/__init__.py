import os

# Custom
from networks._registry import collect_builders, get_builder

_builders = collect_builders(__name__, os.path.dirname(__file__))


def list_classifiers() -> list[str]:
    """利用可能な分類器名 (= このパッケージ直下のモジュール名) を返す。"""
    return sorted(_builders)


def build(model_name: str, *args, **kwargs):
    """分類器を名前で組み立てる。"""
    return get_builder(_builders, model_name, "classifier")(*args, **kwargs)
