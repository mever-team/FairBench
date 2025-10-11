import random
import json
import os


class LLMDatasetGenerator:
    def __init__(self):
        self.attrs = list()

    def __setitem__(self, key, values):
        self.attrs.append((key, values))

    def keys(self):
        return [attr for attr, _ in self.attrs]

    def generate(self):
        values = [random.choice(values) for _, values in self.attrs]
        description = " ".join(values)
        return description, values


def _progress_bar(i, n, message):
    percent = int((i / n) * 100)
    filled = int((i / n) * 20)  # 20 total spaces
    bar = "#" * filled + "-" * (20 - filled)
    print(f"\r{message} {percent:3d}% [{bar}]", end="")


def simplequestions(
    model,
    attributes: dict | LLMDatasetGenerator,
    query_prototype,
    cache="dataset.json",
    n=1000,
    overwrite=False,
):
    if os.path.exists(cache) and not overwrite:
        with open(cache, "r") as file:
            dataset = json.load(file)
        return dataset, dataset["reply"]

    assert (
        "{demographic}" in query_prototype
    ), "The query prototype must contain a `{demographic}` substring"
    if isinstance(attributes, dict):
        attr = LLMDatasetGenerator()
        for k, v in attributes.items():
            assert isinstance(v, tuple) or isinstance(v, list) or isinstance(v, set), (
                "Only lists, tuples, or sets allowed as attribute values. Found in attribute: "
                + str(k)
            )
            attr[k] = tuple(v)
        attributes = attr
    assert isinstance(attributes, LLMDatasetGenerator), (
        "Only dict from demographic attribute str to value lists"
        "or an LLMDatasetGenerator are allowed as attributes"
    )
    dataset = {attr: list() for attr in attributes.keys()}

    assert "query" not in dataset, "Cannot have an attribute called `query`"
    assert "reply" not in dataset, "Cannot have an attribute called `reply`"
    dataset["query"] = list()
    dataset["reply"] = list()

    for i in range(n):
        _progress_bar(i, n, "Creating query variations: ")
        description, values = attributes.generate()
        query = query_prototype.replace("{demographic}", description)
        reply = model(query)[len(query) :].strip()
        for attr, value in zip(attributes.keys(), values):
            dataset[attr].append(value)
        dataset["query"].append(query)
        dataset["reply"].append(reply)
        # print(description, reply)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as file:
        json.dump(dataset, file, indent=2)

    return dataset, dataset["reply"]
