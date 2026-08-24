class ToDict:
    def direct_show(self, value, depth=0):
        assert (
            depth == 0
        ), "dictionary conversion should not specify a depth (it considers the whole depth)"
        return value.to_dict()
