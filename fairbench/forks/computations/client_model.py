class _NoClient:  # emulates the dask.distributed.Client api with centralized computations
    def submit(
        self,
        method,
        *args,
        workers=None,
        allow_other_workers=True,
        pure=False,
        **kwargs,
    ):
        assert allow_other_workers
        assert not pure
        return method(*args, **kwargs)


_client = _NoClient()


def distributed(*args, **kwargs):
    global _client
    from dask.distributed import Client

    if isinstance(_client, Client):
        _client.close()
    _client = Client(*args, **kwargs)


def client():
    global _client
    return _client
