from itertools import islice


def grouper(n: int, iterable: iter):
    while True:
        chunk = tuple(islice(iterable, n))
        if not chunk:
            return
        yield chunk


def cycle(iterable: iter):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)
