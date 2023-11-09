class ExplainableError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.explain = message
        self.value = float("NaN")
        self.distribution = None
        self.desc = None

    def __float__(self):
        return self.value

    def __int__(self):
        return self.value

    def __str__(self):
        return "---"
