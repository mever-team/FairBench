from fairbench.forks.fork import Fork, Forklike
from fairbench.forks.explanation import tofloat
import sys
from bokeh.transform import dodge



def interactive(report):  # pragma: no cover
    try:
        import bokeh
    except ImportError:
        sys.stderr.write("install bokeh for interactive visualization in the browser.")
        return
    from bokeh.models import ColumnDataSource, Select, Range1d, Button, Div, FactorRange
    from bokeh.plotting import figure
    from bokeh.layouts import column, row
    from bokeh.server.server import Server
    from bokeh.application import Application
    from bokeh.application.handlers import FunctionHandler
    import webbrowser
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Category20

    plot = figure(x_range=["1", "2"], width=1400)
    select_branch = Select(value="---", options=["---"]+list(report.branches().keys()))
    explain = Button(label="Explain")
    back = Button(label="Back")
    select_branch.sizing_mode = "stretch_width"
    explain.sizing_mode = "stretch_width"
    back.sizing_mode = "stretch_width"
    previous = [report]
    previous_title = ["report"]
    label = Div()

    def _asdict(obj):
        if isinstance(obj, Fork):
            return obj.branches()
        return obj

    def update_plot(doc):
        branches = _asdict(previous[-1])
        back.visible = len(previous) > 1 or select_branch.value != "---"
        plot.renderers = []  # clear plot
        selected_branch = select_branch.value
        if selected_branch == "---":
            _source = dict()
            if isinstance(branches[list(branches.keys())[0]], Fork):
                for branch in branches.keys():
                    for k, v in _asdict(branches[branch]).items():
                        if k not in _source:
                            _source[k] = list()
                        _source[k].append(branch)
                values = [tofloat(_asdict(branches[branch])[metric]) for metric in _source for branch in _source[metric]]
            else:
                for branch in branches.keys():
                    for k, v in _asdict(branches[branch]).items():
                        if branch not in _source:
                            _source[branch] = list()
                        _source[branch].append(k)
                values = [tofloat(_asdict(branches[branch])[metric]) for branch in _source for metric in _source[branch]]

            explain.visible = False
            label.text = f"<h1>{'.'.join(previous_title)}</h1>"
            plot.xgrid.grid_line_color = None
            keys = [(metric, branch) for metric in _source for branch in _source[metric]]
            source = ColumnDataSource(data=dict(keys=keys, values=values))
            plot.x_range.factors = keys
            plot.x_range.range_padding = 0.1
            plot.vbar(x='keys', top='values',
                      width=0.9,
                      source=source,
                      #legend_field='keys',
                      line_color='white',
                      fill_color=factor_cmap('keys', palette=[Category20[20][i] for metric in _source for i, branch in enumerate(_source[metric])], factors=keys)
                      )
        else:
            plot_data = branches[selected_branch]
            explain.visible = hasattr(plot_data, "explain")
            if isinstance(plot_data, Fork):
                plot_data = plot_data.branches()
            keys = list(plot_data.keys())
            values = [tofloat(value) for value in plot_data.values()]
            source = ColumnDataSource(data=dict(keys=keys, values=values))
            label.text = f"<h1>{'.'.join(previous_title)}.<em>{selected_branch}</em></h1>"
            plot.xgrid.grid_line_color = None
            plot.x_range.factors = keys
            #plot.y_range.update(start=0, end=max(1, max(values)), bounds=(0, None))
            plot.vbar(x='keys', top='values',
                      width=0.6,
                      source=source,
                      #legend_field='keys',
                      line_color='white',
                      fill_color=factor_cmap('keys', palette=Category20[20], factors=keys)
                      )

    def explain_button(doc):
        previous_title.append(select_branch.value)
        previous_title.append("explain")
        branches = _asdict(previous[-1])
        branch = branches[select_branch.value]
        previous.append(branch.explain)
        branches = _asdict(previous[-1])
        select_branch.options = ["---"] + list(branches.keys())
        select_branch.value = "---"
        update_plot(doc)

    def back_button(doc):
        if select_branch.value != "---":
            select_branch.value = "---"
            update_plot(doc)
            return
        previous_title.pop()
        previous_title.pop()
        previous.pop()
        branches = previous[-1].branches() if isinstance(previous[-1], Fork) else previous[-1]
        select_branch.options = ["---"]+list(branches.keys())
        select_branch.value = "---"
        update_plot(doc)

    def modify_doc(doc):
        doc.title = "FairBench report"
        select_branch.on_change('value', lambda attr, old, new: update_plot(doc))
        explain.on_click(explain_button)
        back.on_click(back_button)
        update_plot(doc)
        controls = column(select_branch, back, explain)
        doc.add_root(column(label, row(controls, plot)))

    app = Application(FunctionHandler(modify_doc))
    server = Server({'/': app}, num_procs=1)
    server.start()
    address = server.address.split(':')[0] if server.address else "localhost"
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


