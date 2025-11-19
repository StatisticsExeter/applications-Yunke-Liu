def sum_list(values):
    """Return the sum of a list of numbers."""
    return sum(values)


def max_value(values):
    """Return the maximum value in a non-empty list of numbers."""
    if not values:
        raise ValueError("values must not be empty")
    return max(values)


def reverse_string(s):
    """Return the reverse of a string."""
    return s[::-1]


def filter_even(values):
    """Return a list of even numbers from the input list."""
    return [v for v in values if v % 2 == 0]


def get_fifth_row(df):
    """Return the 5th row (index 4) of a DataFrame.
    Raise IndexError if the dataframe has fewer than 5 rows.
    """
    if len(df) < 5:
        raise IndexError("DataFrame has fewer than 5 rows")
    return df.iloc[4]


def column_mean(df, column):
    """Return the mean of the specified column.
    Raise KeyError if the column does not exist.
    Return nan if the column is empty.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    series = df[column]
    if len(series) == 0:
        return float('nan')

    return series.mean()


def lookup_key(d, key):
    """Return the value for `key` in dictionary `d`.
    Return None if key does not exist.
    """
    return d.get(key, None)


def count_occurrences(values):
    """Return a dictionary counting occurrences of each element in a list."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def list_to_string(values):
    """Convert a list of strings into a comma-separated string."""
    return ",".join(values)


def parse_date(date_str):
    """Parse a YYYY-MM-DD string into a datetime.date object.
    Raise ValueError if the string is invalid.
    """
    import datetime
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Invalid date format")
