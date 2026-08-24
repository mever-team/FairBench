import json


class ToJson:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def direct_show(self, value, depth=None):
        assert (
            depth is None
        ), "dictionary conversion should not specify a depth (it considers the whole depth)"
        return json.dumps(value.to_dict(), **self.kwargs)
