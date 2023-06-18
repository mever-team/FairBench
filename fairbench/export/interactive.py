from fairbench.forks.fork import Fork, Forklike
from fairbench.forks.explanation import tofloat, ExplainableError
import sys


def _in_ipynb():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def interactive(report):  # pragma: no cover
    try:
        import bokeh
    except ImportError:
        sys.stderr.write("install bokeh for interactive visualization in the browser.")
        return
    from bokeh.models import (
        ColumnDataSource,
        Select,
        Range1d,
        Button,
        Div,
        FactorRange,
        RadioButtonGroup,
    )
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers import FunctionHandler
    import webbrowser
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Category20
    from bokeh.core.validation import silence
    from bokeh.core.validation.warnings import MISSING_RENDERERS

    silence(MISSING_RENDERERS, True)

    def modify_doc(doc):

        plot = figure(x_range=["1", "2"], width=1200)
        select_branch = RadioButtonGroup(
            labels=["ALL"] + list(report.branches().keys()), active=0
        )
        select_view = Select(
            value="Branches", options=["Branches", "Entries"], width=100, height=30
        )
        explain = Button(label="Explain", width=100, button_type="success")
        back = Button(label="Back", button_type="danger")
        explain.sizing_mode = "stretch_width"
        back.sizing_mode = "stretch_width"
        previous = [report]
        previous_title = ["report"]
        label = Div()

        def _asdict(obj):
            if isinstance(obj, Fork):
                obj = obj.branches()
            if isinstance(obj, dict):
                obj = {
                    k: v
                    for k, v in obj.items()
                    if not isinstance(v, ExplainableError) and not isinstance(v, str)
                }
            return obj

        def _depth(obj):
            if isinstance(obj, Fork):
                return max(_depth(x) for x in obj.branches().values()) + 1
            if isinstance(obj, dict):
                return max(_depth(x) for x in obj.values()) + 1
            return 0

        def _update_branches():
            branches = _asdict(previous[-1])
            if _depth(previous[-1]) <= 1:
                select_branch.labels = ["ALL"] + list(branches.keys())
                return
            _source = dict()
            if (select_view.value == "Branches") != isinstance(previous[-1], Fork):
                for branch in branches.keys():
                    for k, v in _asdict(branches[branch]).items():
                        if k not in _source:
                            _source[k] = list()
                        _source[k].append(branch)
            else:
                _source = branches
            select_branch.labels = ["ALL"] + list(_source.keys())

        def update_plot(doc):
            branches = _asdict(previous[-1])
            back.visible = len(previous) > 1 or select_branch.active != 0
            plot.renderers = []  # clear plot
            selected_branch = select_branch.labels[select_branch.active]
            plot.x_range.factors = []
            select_view.disabled = not True
            if selected_branch == "ALL":
                plot.xgrid.grid_line_color = None
                explain.visible = False
                label.text = f"<h1>{'.'.join(previous_title)}</h1>Select a branch or entry to focus and explain it."
                _source = dict()
                if (select_view.value == "Branches") != isinstance(previous[-1], Fork):
                    for branch in branches.keys():
                        for k, v in _asdict(branches[branch]).items():
                            if k not in _source:
                                _source[k] = list()
                            _source[k].append(branch)
                    try:
                        values = [
                            tofloat(_asdict(branches[branch])[metric])
                            for metric in _source
                            for branch in _source[metric]
                        ]
                    except TypeError:
                        return
                else:
                    for branch in branches.keys():
                        for k, v in _asdict(branches[branch]).items():
                            if branch not in _source:
                                _source[branch] = list()
                            _source[branch].append(k)
                    try:
                        values = [
                            tofloat(_asdict(branches[branch])[metric])
                            for branch in _source
                            for metric in _source[branch]
                        ]
                    except TypeError:
                        return
                keys = [
                    (metric, branch) for metric in _source for branch in _source[metric]
                ]
                plot.x_range.factors = keys
                source = ColumnDataSource(data=dict(keys=keys, values=values))
                plot.x_range.range_padding = 0.1
                plot.vbar(
                    x="keys",
                    top="values",
                    width=0.9,
                    source=source,
                    # legend_field='keys',
                    line_color="white",
                    fill_color=factor_cmap(
                        "keys",
                        palette=[
                            Category20[20][i]
                            for metric in _source
                            for i, branch in enumerate(_source[metric])
                        ],
                        factors=keys,
                    ),
                )
                select_view.disabled = not True
            else:
                select_view.disabled = not False
                label.text = f"<h1>{'.'.join(previous_title)}.<em>{selected_branch}</em></h1>Select ALL to switch between branch and entry views."
                plot_data = (
                    branches[selected_branch]
                    if selected_branch in branches
                    else getattr(previous[-1], selected_branch)
                )
                explain.visible = hasattr(plot_data, "explain")
                plot_data = _asdict(plot_data)
                keys = list(plot_data.keys())
                plot.x_range.factors = keys
                if isinstance(plot_data, Fork):
                    plot_data = plot_data.branches()
                try:
                    values = [tofloat(value) for value in plot_data.values()]
                except TypeError:
                    return
                source = ColumnDataSource(data=dict(keys=keys, values=values))
                plot.xgrid.grid_line_color = None
                # plot.y_range.update(start=0, end=max(1, max(values)), bounds=(0, None))
                plot.vbar(
                    x="keys",
                    top="values",
                    width=0.6,
                    source=source,
                    # legend_field='keys',
                    line_color="white",
                    fill_color=factor_cmap(
                        "keys", palette=Category20[20], factors=keys
                    ),
                )

        def explain_button(doc):
            selected_branch = select_branch.labels[select_branch.active]
            previous_title.append(selected_branch)
            previous_title.append("explain")
            branches = _asdict(previous[-1])
            selected_branch = select_branch.labels[select_branch.active]
            branch = (
                branches[selected_branch]
                if selected_branch in branches
                else getattr(previous[-1], selected_branch)
            )
            previous.append(branch.explain)
            # branches = _asdict(previous[-1])
            # select_branch.labels = ["ALL"] + list(branches.keys())
            _update_branches()
            select_branch.active = 0
            update_plot(doc)

        def back_button(doc):
            if select_branch.labels[select_branch.active] != "ALL":
                select_branch.active = 0
                update_plot(doc)
                return
            if previous_title[-1] == "explain":
                previous_title.pop()
                branch_name = previous_title[-1]
                previous_title.pop()
            else:
                branch_name = "ALL"
                previous_title.pop()
            previous.pop()
            _update_branches()
            for i, label in enumerate(select_branch.labels):
                if label == branch_name:
                    select_branch.active = i
            # if select_branch.active in branches and _depth(branches[select_branch.active ]) > 1:
            #    previous_title.pop()
            update_plot(doc)

        def update_value(doc):
            branches = _asdict(previous[-1])
            selected_branch = select_branch.labels[select_branch.active]
            if (
                selected_branch != "ALL"
                and _depth(
                    branches[selected_branch]
                    if selected_branch in branches
                    else getattr(previous[-1], selected_branch)
                )
                > 1
            ):
                previous_title.append(selected_branch)
                branch = (
                    branches[selected_branch]
                    if selected_branch in branches
                    else getattr(previous[-1], selected_branch)
                )
                previous.append(branch)
                _update_branches()
                select_branch.active = 0
            update_plot(doc)

        def update_view(doc):
            _update_branches()
            update_plot(doc)

        doc.title = "FairBench report"
        _update_branches()
        select_branch.on_change("active", lambda attr, old, new: update_value(doc))
        select_view.on_change("value", lambda attr, old, new: update_view(doc))
        explain.on_click(explain_button)
        back.on_click(back_button)
        update_plot(doc)
        controls = row(select_view, select_branch, back, explain)
        doc.add_root(column(label, controls, plot))

    app = Application(FunctionHandler(modify_doc))
    if _in_ipynb():
        from bokeh.io import show
        from bokeh.plotting import output_notebook

        output_notebook()
        show(app)
        return
    server = Server({"/": app}, num_procs=1)
    server.start()
    address = server.address.split(":")[0] if server.address else "localhost"
    server_url = f"http://{address}:{server.port}/"
    print(f"Bokeh server started at {server_url}")

    try:
        webbrowser.open_new(server_url)
        import asyncio

        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        print("Bokeh server stopped.")
    finally:
        server.stop()

    """show(app)
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()"""
