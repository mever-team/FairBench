from fairbench.forks.fork import Fork, Forklike
from fairbench.forks.explanation import tofloat, ExplainableError


def clean(fork):
    if isinstance(fork, Fork):
        branches = {k: clean(v) for k, v in fork.branches().items()}
        branches = {k: v for k, v in branches.items() if v is not None}
        if not branches:
            return None
        return Fork(branches)
    if isinstance(fork, Forklike):
        branches = {k: clean(v) for k, v in fork.items()}
        branches = {k: v for k, v in branches.items() if v is not None}
        if not branches:
            return None
        return Forklike(branches)
    if isinstance(fork, ExplainableError):
        return None
    return fork


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


def interactive(report, name="report", width=1200):  # pragma: no cover
    from bokeh.models import (
        ColumnDataSource,
        Select,
        Range1d,
        Button,
        Div,
        FactorRange,
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

    silence(MISSING_RENDERERS, True)

    def modify_doc(doc):
        plot = figure(x_range=["1", "2"], width=width)
        plot.add_tools(HoverTool(tooltips=[("Name", "@keys"), ("Value", "@values")]))
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
            return getattr(Forklike(branches), selected_branch)

        def _depth(obj):
            if obj is None:
                return 0
            if isinstance(obj, Fork):
                return max(_depth(x) for x in obj.branches().values()) + 1
            if isinstance(obj, dict):
                if obj.values():
                    return max(_depth(x) for x in obj.values()) + 1
            return 0

        def _update_branches():
            branches = _asdict(previous[-1])
            if isinstance(previous[-1], Forklike) or isinstance(previous[-1], Fork):
                val = list(branches.values())[0].role()
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
            plot.x_range.factors = []
            select_view.disabled = not True
            plot.title.text = "" if selected_branch == "ALL" else selected_branch
            plot.title_location = "above"
            if selected_branch == "ALL":
                plot.xgrid.grid_line_color = None
                explain.visible = False
                label.text = f"<h1>{'.'.join([t for t in previous_title if t!='ALL'])}</h1>Select a branch or entry to focus and explain it."
                _source = dict()
                if (select_view.value == select_view.options[0]) != isinstance(
                    previous[-1], Fork
                ):
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
                label.text = f"<h1>{'.'.join([t for t in previous_title if t!='ALL'])}.<em>{selected_branch}</em></h1>Select ALL to switch between branch and entry views."
                plot_data = _branch(selected_branch)
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
            selected_branch = select_branch.labels[select_branch.active]
            branch = (
                previous[-1] if selected_branch == "ALL" else _branch(selected_branch)
            )
            previous.append(clean(branch.explain))
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
            selected_branch = select_branch.labels[select_branch.active]
            if selected_branch != "ALL" and _depth(_branch(selected_branch)) > 1:
                previous_title.append(selected_branch)
                branch = _branch(selected_branch)
                previous.append(clean(branch))
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
