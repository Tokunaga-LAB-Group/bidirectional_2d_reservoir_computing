# Custom
from networks import modules


# 特徴抽出を一切行わず、画像を 1 本のベクトルに潰してそのまま線形読み出しに渡す
#
# NOTE: 全条件の下限となる対照。読み出し (リッジ回帰) だけで到達できる精度を示すので、
#       他の条件の精度からこの値を引いたものが「特徴抽出器が足した分」になる。
#       生画素に対する多クラス線形回帰そのものであり、固定ランダム重みを 1 つも持たない
#
# NOTE: fcl は固定ランダム射影 + tanh を挟むため Extreme Learning Machine に相当し、
#       「特徴抽出器あり」の条件である。こちらはその射影すら持たない
#
# NOTE: feature_dim は入力次元 (C * H * W) で固定される。units を受け取らないため
#       D 走査には参加せず、表では水平な基準線になる
class RawClassifier(modules.Classifier):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        n_layer: int | None = None,
        seed: int | list[int] = 0,
    ):
        C, H, W = input_shape

        # NOTE: n_layer / seed は他の classifier と引数を揃えるために受け取るだけで使わない。
        #       積む層も引く乱数も無いため
        if n_layer is not None and int(n_layer) != 1:
            raise ValueError(f"raw has no layers to stack, got n_layer={n_layer}.")

        super().__init__(feature_dim=C * H * W, num_classes=num_classes)

    # (B, C, H, W) -> (B, feature_dim)
    def features(self, images):
        return images.flatten(1)  # (B, C * H * W)


def build(input_shape, num_classes, **kwargs):
    """networks.build_classifier("raw", ...) のエントリポイント。

    Args:
        input_shape: 入力画像の形状 (C, H, W)。
        num_classes: クラス数。
    """
    return RawClassifier(input_shape, num_classes, **kwargs)
