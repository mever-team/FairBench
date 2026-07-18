import fairbench as fb
from fairbench.bench.wizard import BENCH_REGISTRY

BENCH_REGISTRY["custom"] = {
    "label": "Custom experiment (declared programmatically)",
    "func": lambda: fb.bench.tabular.compas(),
}

if __name__ == "__main__":
    fb.bench.serve_wizard(port=8000)
