import fairbench
from fairbench.v1.core import Fork, DotDict
from fairbench.v1.core.explanation.base import tofloat
from fairbench.v1.core import ExplainableError


def _clean(fork):  # pragma: no cover
    if isinstance(fork, Fork):
        branches = {k: _clean(v) for k, v in fork.branches().items()}
        branches = {k: v for k, v in branches.items() if v is not None}
        if not branches:
            return None
        return Fork(branches)
    if isinstance(fork, DotDict):
        branches = {k: _clean(v) for k, v in fork.items()}
        branches = {k: v for k, v in branches.items() if v is not None}
        if not branches:
            return None
        return DotDict(branches)
    if isinstance(fork, ExplainableError):
        return None
    return fork


def _in_jupyter():  # pragma: no cover
    """
    Checks whether current code runs within a Jupyter notebook.
    :return: True if within Jupyter, False otherwise
    """
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


def interactive(
    report,
    name="report",
    width=800,
    height=400,
    spacing=None,
    horizontal=True,
    port=8888,
    browser=None,
):  # pragma: no cover
    """
    Creates an interactive visualization over a fairness report.
    :param report: A fairness report.
    :param name: Default is 'report'.
    :param width: The minimum width of interactive plot screens.
    :param height: The minimum height of interactive plot screens.
    :param spacing: The minimum spacing between bars of bar plots. If None, internally set to 30 for horizontal mode and 80 otherwise.
    :param horizontal: Whether bar plots should be horizontally or vertically aligned.
    """
    try:
        import bokeh
    except ImportError:
        print(
            "Interactive visualization dependencies are installed by `pip install fairbench[interactive]`."
            "\nFor now, `fairbench.interactive_html` is used as a fallback."
        )
        from fairbench.v1.export.interactive_html import interactive_html

        return interactive_html(report, name=name)
    from bokeh.models import (
        ColumnDataSource,
        Select,
        Button,
        Div,
        HoverTool,
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

    if spacing is None:
        spacing = 30 if horizontal else 80

    silence(MISSING_RENDERERS, True)
    order = (lambda x: x) if not horizontal else reversed

    def modify_doc(doc):
        if horizontal:
            plot = figure(
                y_range=["1", "2"], width=width, height=height, x_axis_location="above"
            )
        else:
            plot = figure(x_range=["1", "2"], width=width, height=height)
        plot.add_tools(HoverTool(tooltips=[("Name", "@keys"), ("Value", "@values")]))
        curves = figure(width=width, height=height)
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
        previous_title = [name]
        label = Div()
        cannot_observe = Div()
        cannot_observe.text = (
            "Data too complicated to visualize. Focus on one branch or entry."
        )
        cannot_observe.visible = False

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

        def _branch(selected_branch):
            branches = _asdict(previous[-1])
            if selected_branch in branches:
                return branches[selected_branch]
            return getattr(DotDict(branches), selected_branch)

        def _depth(obj):
            if obj is None:
                return 0
            elif isinstance(obj, Fork):
                branches = obj.branches()
                if branches:
                    return max(_depth(x) for x in branches.values()) + 1
            elif isinstance(obj, dict):
                if obj:
                    return max(_depth(x) for x in obj.values()) + 1
            return 0

        def _update_branches():
            branches = _asdict(previous[-1])
            if not branches:
                select_branch.labels = ["ALL"]
                return

            if isinstance(previous[-1], DotDict) or isinstance(previous[-1], Fork):
                val = (
                    ""
                    if not hasattr(list(branches.values())[0], "role")
                    else list(branches.values())[0].role()
                )
                prev_selection = select_view.value
                options = ["Branch", "Entries" if val is None else val]
                select_view.options = options
                if prev_selection not in options:
                    select_view.value = options[0]

            if _depth(previous[-1]) <= 1:
                select_branch.labels = ["ALL"] + list(branches.keys())
                return
            _source = dict()
            if (select_view.value == select_view.options[0]) != isinstance(
                previous[-1], Fork
            ):
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
            if horizontal:
                plot.y_range.factors = []
            else:
                plot.x_range.factors = []
            select_view.disabled = not True
            plot.title.text = "" if selected_branch == "ALL" else selected_branch
            plot.title_location = "above"
            plot.visible = True
            curves.visible = False
            cannot_observe.visible = False
            if selected_branch == "ALL":
                if horizontal:
                    plot.ygrid.grid_line_color = None
                else:
                    plot.xgrid.grid_line_color = None
                explain.visible = False
                label.text = f"<h1>{'.'.join([t for t in previous_title if t!='ALL'])}</h1>Select a branch or entry to focus and explain it."
                _source = dict()
                if (select_view.value == select_view.options[0]) != isinstance(
                    previous[-1], Fork
                ):
                    if branches:
                        for branch in branches.keys():
                            for k, v in _asdict(branches[branch]).items():
                                if k not in _source:
                                    _source[k] = list()
                                _source[k].append(branch)
                    try:
                        values = [
                            tofloat(_asdict(branches[branch])[metric])
                            for metric in order(_source)
                            for branch in order(_source[metric])
                        ]
                    except TypeError:
                        cannot_observe.visible = True
                        plot.visible = False
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
                            for branch in order(_source)
                            for metric in order(_source[branch])
                        ]
                    except TypeError:
                        cannot_observe.visible = True
                        plot.visible = False
                        return
                if not _source:
                    pass
                keys = [
                    (metric, branch)
                    for metric in order(_source)
                    for branch in order(_source[metric])
                ]
                if horizontal:
                    plot.height = max(height, spacing * len(keys))
                    plot.y_range.factors = keys
                    plot.y_range.range_padding = 0.1
                else:
                    plot.width = max(width, spacing * len(keys))
                    plot.x_range.factors = keys
                    plot.x_range.range_padding = 0.1
                source = ColumnDataSource(data=dict(keys=keys, values=values))
                if horizontal:
                    plot.hbar(
                        y="keys",
                        right="values",
                        height=0.9,
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
                else:
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
                label.text = f"<h1>{'.'.join([t for t in previous_title if t!='ALL'])}.<em>{selected_branch}</em></h1>Select ALL to switch between branch and entry views."
                plot_data = _branch(selected_branch)
                explain.visible = hasattr(plot_data, "explain")
                plot_data = _asdict(plot_data)
                if hasattr(plot_data, "keys") and (
                    not list(plot_data.keys())
                    or isinstance(
                        plot_data[list(plot_data.keys())[0]], fairbench.ExplanationCurve
                    )
                ):
                    plot.visible = False
                    curves.visible = True
                    curves.renderers = []
                    curves.legend.items = []
                    for i, k in enumerate(plot_data):
                        curve = plot_data[k]
                        curves.line(
                            curve.x,
                            curve.y,
                            line_color=Category20[20][i],
                            line_width=2,
                            legend_label=curve.name + " of " + k,
                        )
                    if plot_data:
                        curves.legend.location = "bottom_right"
                    # curves.legend.title = selected_branch
                    return
                plot_data = {k: plot_data[k] for k in order(plot_data)}
                keys = list(plot_data.keys())
                if horizontal:
                    plot.height = max(height, spacing * len(keys))
                    plot.y_range.factors = keys
                else:
                    plot.width = max(width, spacing * len(keys))
                    plot.x_range.factors = keys
                if isinstance(plot_data, Fork):
                    plot_data = plot_data.branches()
                try:
                    values = [tofloat(value) for value in plot_data.values()]
                except TypeError:
                    return
                source = ColumnDataSource(data=dict(keys=keys, values=values))
                if horizontal:
                    plot.ygrid.grid_line_color = None
                    plot.hbar(
                        y="keys",
                        right="values",
                        height=0.6,
                        source=source,
                        # legend_field='keys',
                        line_color="white",
                        fill_color=factor_cmap(
                            "keys", palette=Category20[20], factors=keys
                        ),
                    )
                else:
                    plot.xgrid.grid_line_color = None
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
                # plot.y_range.update(start=0, end=max(1, max(values)), bounds=(0, None))

        def explain_button(doc):
            selected_branch = select_branch.labels[select_branch.active]
            previous_title.append(selected_branch)
            previous_title.append("explain")
            selected_branch = select_branch.labels[select_branch.active]
            branch = (
                previous[-1] if selected_branch == "ALL" else _branch(selected_branch)
            )
            previous.append(_clean(branch.explain))
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
            select_branch.active = 0
            for i, label in enumerate(select_branch.labels):
                if label == branch_name:
                    select_branch.active = i
            if select_branch.active == 0:
                if select_view.value == select_view.options[0]:
                    select_view.value = select_view.options[1]
                else:
                    select_view.value = select_view.options[0]
                _update_branches()
                for i, label in enumerate(select_branch.labels):
                    if label == branch_name:
                        select_branch.active = i
            # if select_branch.active in branches and _depth(branches[select_branch.active ]) > 1:
            #    previous_title.pop()
            update_plot(doc)

        def update_value(doc):
            selected_branch = (
                select_branch.labels[select_branch.active]
                if select_branch.labels
                else "ALL"
            )
            if selected_branch != "ALL" and _depth(_branch(selected_branch)) > 1:
                previous_title.append(selected_branch)
                branch = _branch(selected_branch)
                previous.append(_clean(branch))
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
        doc.add_root(column(label, controls, plot, curves, cannot_observe))

    app = Application(FunctionHandler(modify_doc))
    if browser is not None and not browser and _in_jupyter():
        from bokeh.io import show
        from bokeh.plotting import output_notebook
        import os

        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = "localhost:" + str(port)
        output_notebook()
        show(app)
        return
    if browser is not None and browser == False:
        raise Exception("Cannot set broswer=False when not using Jupyter,")
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
