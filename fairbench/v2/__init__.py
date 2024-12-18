from fairbench import fallbacks
from fairbench import bench

from fairbench.v2.blocks import *
from fairbench.v2.core import Sensitive, Progress
from fairbench.v2 import core
from fairbench.v2 import export
from fairbench.v2.export import help
from fairbench.v2 import reports
from fairbench.v2 import investigate

from fairbench.v1 import categories, fuzzy


def Dimensions(*args, **kwargs):
    from fairbench.v1 import Fork, tobackend

    ret = Fork(*args, **kwargs)
    ret._branches = {
        k: (tobackend(v) if isinstance(v, list) and not isinstance(v, Fork) else v)
        for k, v in ret._branches.items()
    }
    return ret
