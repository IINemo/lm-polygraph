def flatten(a: list[list] | None) -> list | None:
    if a is None:
        return None
    return [c for b in a for c in b]


def reconstruct(b: list, a: list[list]) -> list[list]:
    assert sum(len(x) for x in a) == len(b)
    last = 0
    b_reconstructed = []
    for x in a:
        b_reconstructed.append(b[last : last + len(x)])
        last += len(x)
    return b_reconstructed
