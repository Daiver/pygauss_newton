import time


def time_fn(function, *args, **kwargs):
    time_start = time.time()
    results = function(*args, **kwargs)
    time_end = time.time()
    return results, time_end - time_start
