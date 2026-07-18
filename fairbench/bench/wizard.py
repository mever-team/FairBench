import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from importlib.resources import files
import fairbench as fb

BENCH_REGISTRY = {
    "tabular.bank": {
        "label": "Bank (tabular)",
        "params": {
            "test_size": {
                "type": "train_percent",
                "default": 50,
                "label": "Training data split",
            }
        },
    },
    "tabular.compas": {
        "label": "COMPAS (tabular)",
        "params": {
            "test_size": {
                "type": "train_percent",
                "default": 50,
                "label": "Training data split",
            },
        },
    },
    "tabular.adult": {
        "label": "Adult Income (tabular)",
        "params": {},
    },
}

REPORT_REGISTRY = {
    "pairwise": {
        "label": "pairwise (compare groups pairwise)",
        "params": {},
    },
    "vsall": {"label": "vsall (compare groups to the total population)", "params": {}},
}

VIS_REGISTRY = {
    "html": {
        "label": "as html",
        "env": lambda: fb.export.Html(filename=None),
    },
    "html table": {
        "label": "as html table",
        "code": "HtmlTable",
        "env": lambda: fb.export.HtmlTable(filename=None),
    },
    "html bars": {
        "label": "as html bars",
        "env": lambda: fb.export.HtmlBars(filename=None),
    },
}

FILTER_REGISTRY = {
    "none": {
        "label": "no filtering",
        "path": None,
        "params": {},
    },
    "stamps": {
        "label": "stamps: a model card with popular named measures and qualitative concerns",
        "path": "investigate.Stamps",
        "params": {},
    },
    "deviations": {
        "label": "deviations: show only large deviations from ideal values",
        "path": "investigate.DeviationsOver",
        "params": {
            "limit": {
                "type": "train_percent",
                "default": 20,
                "label": "Numerical limit",
            }
        },
    },
    "isbias": {
        "label": "biases: show only measures of bias (ideal value is zero)",
        "path": "investigate.IsBias",
        "params": {},
    },
}

STATE = {
    "x": None,
    "y": None,
    "yhat": None,
    "sensitive": None,
    "report": None,
    "html": None,
}

def resolve(root, dotted_path):
    obj = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def coerce(value, ptype):
    if ptype in ("float", "slider_float"):
        return float(value)
    if ptype in ("int", "slider_int"):
        return int(value)
    if ptype == "bool":
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes", "on")
    if ptype == "train_percent":
        # UI shows/sends percentage of *training* data; underlying function
        # wants test_size (fraction held out for testing).
        return round(1 - float(value) / 100, 4)
    return value


def build_params(schema, submitted):
    submitted = submitted or {}
    out = {}
    for name, spec in schema.items():
        if name in submitted:
            out[name] = coerce(submitted[name], spec["type"])
        elif "default" in spec:
            out[name] = coerce(spec["default"], spec["type"])
    return out

def run_bench(body):
    bench_key = body.get("bench")
    if bench_key not in BENCH_REGISTRY:
        raise ValueError(f"Unknown bench '{bench_key}'")
    bench_spec = BENCH_REGISTRY[bench_key]
    schema = bench_spec.get("params", {})
    params = build_params(schema, body.get("params"))
    if "func" in bench_spec:
        func = bench_spec["func"]
    else:
        func = resolve(fb.bench, bench_key)
    x, y, yhat = func(**params)
    STATE["x"] = x
    STATE["y"] = y
    STATE["yhat"] = yhat
    STATE["sensitive"] = None
    STATE["report"] = None
    STATE["html"] = None
    keys = list(x)
    return {
        "summary": {
            "x_keys": len(keys),
            "len_y": len(y),
            "len_yhat": len(yhat),
        },
        "details": {
            "bench": bench_key,
            "params": params,
            "x_key_list": keys,
        },
    }


def get_attrs():
    if STATE["x"] is None:
        raise ValueError("Run a bench step first")
    return {"attrs": [attr for attr in STATE["x"]]}


def run_sensitive(body):
    if STATE["x"] is None:
        raise ValueError("Run a bench step first")

    attrs = body.get("attrs") or []
    if not attrs:
        raise ValueError("Select at least one sensitive attribute")

    intersectional = bool(body.get("intersectional", False))
    strict = bool(body.get("strict", False))
    min_size = int(body.get("min_size", 0) or 0)
    discrete_threshold = int(body.get("discrete_threshold", 10) or 10)

    x = STATE["x"]
    dims = [fb.autotype(discrete_threshold) @ x[attr] for attr in attrs]
    sensitive = fb.Dimensions(*dims)
    if intersectional:
        sensitive = sensitive.intersectional(min_size=min_size)
    if strict:
        sensitive = sensitive.strict()

    STATE["sensitive"] = sensitive
    STATE["report"] = None
    STATE["html"] = None

    branches = list(sensitive.branches().keys())
    return {
        "summary": {
            "attrs": attrs,
            "intersectional": intersectional,
            "strict": strict,
            "discrete_threshold": discrete_threshold,
            "num_groups": len(branches),
        },
        "details": {
            "branches": branches,
            "sum": str(sensitive.sum()),
        },
    }

def run_report(body):
    if STATE["sensitive"] is None:
        raise ValueError("Run the sensitive attribute step first")

    report_key = body.get("report")
    if report_key not in REPORT_REGISTRY:
        raise ValueError(f"Unknown report '{report_key}'")
    schema = REPORT_REGISTRY[report_key]["params"]
    params = build_params(schema, body.get("params"))
    depth = int(body.get("depth", 0))

    vis_key = body.get("visualization")
    if vis_key not in VIS_REGISTRY:
        raise ValueError(f"Unknown visualization strategy '{vis_key}'")

    filter_key = body.get("filter", "none")
    if filter_key not in FILTER_REGISTRY:
        raise ValueError(f"Unknown filter '{filter_key}'")

    func = resolve(fb.reports, report_key)
    report = func(
        predictions=STATE["yhat"],
        labels=STATE["y"],
        sensitive=STATE["sensitive"],
        **params,
    )

    filter_spec = FILTER_REGISTRY[filter_key]
    if filter_spec["path"]:
        filter_params = build_params(filter_spec["params"], body.get("filter_params"))
        filter_func = resolve(fb, filter_spec["path"])
        report = report.filter(filter_func(**filter_params))

    STATE["report"] = report

    env = VIS_REGISTRY[vis_key]["env"]()
    html_text = report.show(env=env, depth=depth)
    STATE["html"] = html_text

    return {
        "summary": {
            "report": report_key,
            "filter": filter_key,
            "visualization": vis_key,
            "depth": depth,
        }
    }

def get_schemas():
    return {
        "benches": {bench: {k: v for k,v in BENCH_REGISTRY[bench].items() if not callable(v)} for bench in BENCH_REGISTRY},
        "reports": REPORT_REGISTRY,
        "filters": FILTER_REGISTRY,
        "visualizations": {k: {"label": v["label"]} for k, v in VIS_REGISTRY.items()},
    }

ROUTES_GET = {
    "/api/schemas": lambda: get_schemas(),
    "/api/attrs": lambda: get_attrs(),
}
ROUTES_POST = {
    "/api/bench/run": run_bench,
    "/api/sensitive/run": run_sensitive,
    "/api/report/run": run_report,
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            html = (
                files("fairbench.bench")
                .joinpath("wizard_index.html")
                .read_text(encoding="utf-8")
            )
            self._send_html(html)
            return

        if path == "/api/visualization":
            html = (
                STATE["html"]
                or "<p style='font-family:sans-serif;color:#888'>No report generated yet.</p>"
            )
            self._send_html(html)
            return

        if path in ROUTES_GET:
            try:
                self._send_json(ROUTES_GET[path]())
            except Exception as e:
                self._send_json(
                    {"error": str(e), "traceback": traceback.format_exc()}, status=400
                )
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body"}, status=400)
            return

        if path in ROUTES_POST:
            try:
                self._send_json(ROUTES_POST[path](body))
            except Exception as e:
                self._send_json(
                    {"error": str(e), "traceback": traceback.format_exc()}, status=400
                )
            return

        self._send_json({"error": "Not found"}, status=404)


def serve_wizard(port=8000):
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving on http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve_wizard(port=8000)
