def is_in_jupyter():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in (
            "ZMQInteractiveShell",
            "Shell",
        )
    except Exception:
        return False


def display_or_open_html(path, html):
    if is_in_jupyter():
        from IPython.display import HTML, display

        display(HTML(html))
    else:
        import webbrowser

        webbrowser.open(path)
