import importlib
import pkgutil

# pkgutil:
## パッケージ/インポートシステムを操作するためのユーティリティモジュール
## プラグイン探索のiter_modules/walk_packagesは今も使われる

# myapp/                  ← パッケージ
# ├── __init__.py
# ├── config.py           ← モジュール
# ├── utils.py            ← モジュール
# └── models/             ← サブパッケージ（myapp から見て）
#     ├── __init__.py
#     ├── resnet.py       ← モジュール
#     └── heads/          ← models のサブパッケージ
#         ├── __init__.py
#         └── linear.py


def collect_builders(package_name: str, package_dir: str) -> dict:
    """パッケージ直下の各モジュールから build 関数を集めてレジストリを作る。"""
    builders = {}

    # package_dir以下を走査
    for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
        # サブパッケージと非公開モジュールはモデル定義とみなさない
        if is_pkg or module_name.startswith("_"):
            continue

        module = importlib.import_module(f"{package_name}.{module_name}")

        if hasattr(module, "build"):
            builders[module_name] = module.build

    return builders


def get_builder(builders: dict, model_name: str, kind: str):
    """レジストリから build 関数を引く。

    NOTE: builders[model_name] を try/except KeyError で囲むと build 関数の内部で発生した
    KeyError まで "Unknown model" にすり替わるため、事前に存在確認する。
    """
    if model_name not in builders:
        raise ValueError(f"Unknown {kind}: {model_name!r}. Available: {sorted(builders)}")

    return builders[model_name]
