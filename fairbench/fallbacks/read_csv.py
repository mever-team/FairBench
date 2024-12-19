import random
import numpy as np


class FairBenchCSVColumn:
    def __init__(self, data, name=None):
        self._data = np.array(data)
        self.name = name

    @property
    def values(self):
        """Returns the column data as a numpy array."""
        return self._data

    def __iter__(self):
        return iter(self._data)

    def __eq__(self, other):
        return FairBenchCSVColumn(self._data == other)

    def __repr__(self):
        return f"FairBenchCSVColumn(name={self.name}, values={self.values})"


class FairBenchCSV:
    def __init__(self, data):
        """Initializes FairBenchCSV with a dictionary of columns, and concatenates the columns along axis=1."""
        self.columns = {
            col: FairBenchCSVColumn(values, name=col) for col, values in data.items()
        }
        self.column_names = list(data.keys())
        self.values = np.column_stack([col.values for col in self.columns.values()])

    def get_column(self, column_name):
        return self.columns[column_name]

    def __getitem__(self, column_name):
        return self.get_column(column_name)


def get_dummies(col):
    """Creates a one-hot encoding for a FairBenchCSVColumn."""
    unique_values = set(col.values)
    dummy_data = {f"{col.name}_{val}": [] for val in unique_values}

    for value in col.values:
        for dummy_col in dummy_data:
            dummy_data[dummy_col].append(1 if dummy_col.endswith(f"_{value}") else 0)

    return FairBenchCSV(dummy_data)


def concat(cols, axis=1):
    assert axis == 1, "FairBench's concat fallback can only be used to join columns"
    combined_data = {}
    for col in cols:
        if isinstance(col, FairBenchCSV):
            combined_data.update(
                {
                    col_name: col.get_column(col_name).values.tolist()
                    for col_name in col.columns
                }
            )
        elif isinstance(col, FairBenchCSVColumn):
            combined_data[col.name] = col.values.tolist()
        else:
            raise TypeError(
                "FairBench's concat fallback function only accepts FairBenchCSV or FairBenchCSVColumn instances."
            )
    return FairBenchCSV(combined_data)


def convert_to_number(value):
    if value and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    try:
        return int(value) if value.isdigit() else float(value)
    except ValueError:
        return value


def read_csv(
    filepath,
    delimiter=",",
    header=None,
    skipinitialspace=False,
    skiprows=None,
):
    skiprows = {} if skiprows is None else set(skiprows)

    with open(filepath, "r") as file:
        data = None
        for idx, line in enumerate(file):
            if idx in skiprows:
                continue
            line = line[:-1]
            if data is None:
                first_line = line
                first_line = [
                    col.strip() if skipinitialspace else col
                    for col in first_line.split(delimiter)
                ]
                if header == "infer":
                    # Check if the first line looks like headers (contains any non-numeric strings)
                    if '"' in first_line:
                        headers = [
                            col[1:-1] if col[0] == '"' and col[-1] == '"' else col
                            for col in first_line
                        ]
                        data = {header: list() for header in headers}
                    else:
                        headers = [i for i in range(len(first_line))]
                        data = {
                            header: [convert_to_number(value)]
                            for header, value in zip(headers, first_line)
                        }
                elif header == 0:
                    headers = [
                        col[1:-1] if col[0] == '"' and col[-1] == '"' else col
                        for col in first_line
                    ]
                    data = {header: list() for header in headers}
                elif header is None:
                    headers = [i for i in range(len(first_line))]
                    data = {
                        header: [convert_to_number(value)]
                        for header, value in zip(headers, first_line)
                    }
                else:
                    raise AssertionError(
                        "The header argument can only be 'infer', 0, or None"
                    )
                continue

            row = line.strip()
            row = [
                value.strip() if skipinitialspace else value
                for value in row.split(delimiter)
            ]
            if len(row) != len(headers):  # TODO: consider emulating on_bad_lines
                continue
            for i, value in enumerate(row):
                data[headers[i]].append(convert_to_number(value))

    return {k: FairBenchCSVColumn(v, name=k) for k, v in data.items()}


def train_test_split(data, test_size=0.25, random_state=None):
    """Splits data dictionary into training and testing sets based on the specified test_size."""
    if random_state is not None:
        random.seed(random_state)

    train_data, test_data = {}, {}
    num_rows = len(
        next(iter(data.values())).values
    )  # Assume all columns have the same number of rows
    indices = list(range(num_rows))
    random.shuffle(indices)

    split_idx = int(num_rows * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    for column_name, column in data.items():
        train_data[column_name] = [column._data[i] for i in train_indices]
        test_data[column_name] = [column._data[i] for i in test_indices]

    return {k: FairBenchCSVColumn(v, name=k) for k, v in train_data.items()}, {
        k: FairBenchCSVColumn(v, name=k) for k, v in test_data.items()
    }
