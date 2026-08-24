import json


class ToJson:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def direct_show(self, value, depth=0):
        assert (
            depth == 0
        ), "dictionary conversion should not specify a depth (it considers the whole depth)"
        return json.dumps(value.to_dict(), **self.kwargs)
