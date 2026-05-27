from moat_ovha.data.operator_zoo import OperatorZoo


def make_fourier_task(seed: int = 0, split: str = "train", context_size: int = 4, resolution: int = 16):
    return OperatorZoo(seed=seed).make_task("fourier", split, context_size, resolution)
