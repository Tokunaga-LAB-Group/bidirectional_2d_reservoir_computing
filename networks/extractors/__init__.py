import os

# Custom
from networks._registry import collect_builders, get_builder

# __name__: networks.extractors
# __file__: networks/extractors/__init__.py
_builders = collect_builders(__name__, os.path.dirname(__file__))


def list_extractors() -> list[str]:
    """利用可能な特徴抽出器名 (= このパッケージ直下のモジュール名) を返す。"""
    return sorted(_builders)


def build(model_name: str, *args, **kwargs):
    """特徴抽出器を名前で組み立てる。"""
    return get_builder(_builders, model_name, "extractor")(*args, **kwargs)
