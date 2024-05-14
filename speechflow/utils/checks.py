import typing as tp
import inspect
import functools
import subprocess

__all__ = ["check_install", "check_ismethod", "check_isfunction"]


def check_install(*args):
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
        return True
    except OSError as e:
        return False


def check_ismethod(method: tp.Callable) -> bool:
    while True:
        if isinstance(method, functools.partial):
            method = method.func
        else:
            break

    return inspect.ismethod(method)


def check_isfunction(method: tp.Callable) -> bool:
    while True:
        if isinstance(method, functools.partial):
            method = method.func
        else:
            break

    return inspect.isfunction(method)
