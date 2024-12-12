import json


class ToJson:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def direct_show(self, value):
        return json.dumps(value.to_dict(), **self.kwargs)
