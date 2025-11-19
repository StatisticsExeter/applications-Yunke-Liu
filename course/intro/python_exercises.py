def sum_list(values):
    """Return the sum of a list of numbers."""
    return sum(values)


def max_value(values):
    """Return the maximum value in a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    return max(values)
