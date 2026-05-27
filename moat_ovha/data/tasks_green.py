from moat_ovha.data.operator_zoo import OperatorZoo


def make_green_task(seed: int = 0, split: str = "train", context_size: int = 4, resolution: int = 16):
    return OperatorZoo(seed=seed).make_task("green", split, context_size, resolution)
