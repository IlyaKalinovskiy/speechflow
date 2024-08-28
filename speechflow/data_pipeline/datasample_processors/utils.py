import random
import typing as tp

from functools import wraps


def check_probability(method: tp.Callable) -> tp.Callable:
    """Decorator for applying augmentation with probability.

    Probability must be in range [0, 1].

    """

    @wraps(method)
    def use_augmentation(*args, **kwargs):
        p = kwargs.get("p", 1.0)
        if (p < 0) or (p > 1):
            raise ValueError(f"probability of applying aug p must be in [0, 1]. Got {p}.")

        if random.random() > p:
            return kwargs["ds"]

        if len(args) == 0:
            return method(**kwargs)
        else:
            return method(args[0], **kwargs)

    return use_augmentation
