from typing import List, Optional, Tuple, Union
import jittor as jt


def randn_tensor(
    shape: Union[Tuple, List],
    seed: Optional[Union[List[int], int]] = None,
    dtype: Optional[str] = None,
):
    batch_size = shape[0]

    if isinstance(seed, list) and len(seed) == 1:
        seed = seed[0]

    if isinstance(seed, list):
        shape = (1,) + shape[1:]
        latents = [
            seed_randn(shape, dtype, seed[i])
            for i in range(batch_size)
        ]
        latents = jt.concat(latents, dim=0)
    else:
        latents = seed_randn(shape, dtype, seed)

    return latents


def seed_randn(shape, dtype, seed: int):
    if seed is not None:
        jt.set_global_seed(seed)
    var = jt.randn(shape, dtype=dtype)
    return var
