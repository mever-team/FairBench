from fairbench.v2.export.formats.html import Html
from threading import Timer


class WebApp(Html):
    def __init__(self, port=8000):
        super().__init__()
        self.port = port

    def _clear_all(self):
        self.contents = ""
        self.chart_count = 0
        self.bars = []
        self.prev_max_level = 0
        self.prev_level = 0
        self.routes = dict()

    def navigation(self, text, routes: dict):
        # if self.prev_max_level != 1:
        #    return self
        # if self.prev_level == 1:
        #    self.contents += "\n" + text + "<br>"
        for key, text in routes.items():
            # if self.prev_max_level == 1:
            #    self.contents += f'<a href="{key}">{text}</a> '
            if key == "explain":
                self.contents += (
                    f'<br><a class="btn btn-success" href="{key}">{text}</a> '
                )
            self.routes[key] = key
        return self

    def show(self):
        from flask import Flask
        import webbrowser

        app = Flask(__name__)

        @app.route("/", methods=["GET"])
        def handle_index():
            self._clear_all()
            self.rerun(None)
            return self._create_text()

        @app.route("/<method>", methods=["GET"])
        def handle_method(method):
            value = self.routes[method]
            self._clear_all()
            self.rerun(value)
            self.contents = (
                '<h2><a href="/" class="text-danger">Clear selection</a></h2>'
                + self.contents
            )
            return self._create_text()

        def open_browser():
            webbrowser.open(f"http://localhost:{self.port}/")

        Timer(1, open_browser).start()
        app.run(debug=False, port=self.port)

    def title(self, text, level=0, link=None):
        if self.bars:
            self._embed_bars()
            self.bars.clear()
        level = level * 2 + 1
        if level > 6:
            level = 6
        if level == 3:
            if self.prev_max_level >= level:
                self.contents += "</div></div>"
            else:
                self.contents += "<br>"
            self.contents += '<div style="width: 400px; float: left;" class="card m-3">'
            self.contents += f'<h{level} class="mt-0 text-white bg-dark p-3 rounded"><a href="{link}" class="text-white">{text}</a></h{level}>'

            self.contents += """
            <div class="card-body">
            """
        elif level <= 1:
            self.contents += f'<h{level} class="text-dark">{text}</h{level}>'
        else:
            self.contents += (
                f'<h{level} class="mt-5 text-dark"><b>{text}</b></h{level}>'
            )
        self.prev_level = level
        self.prev_max_level = max(self.prev_max_level, level)
        return self
