<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>FairBench</title>
<style>
    #output {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        padding: 10px;
        font-family: monospace;
        margin-bottom: 25px;
    }
    .code-block {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        font-family: monospace;
        spellcheck: false;
        margin-bottom: 25px;
        margin-top: 0px;
        font-size: 0.7em;
    }
    .icon-green {
        color: green;
    }
    .icon-blue {
        color: blue;
    }
    /* Adjust the CodeMirror styling to match your theme preferences */
    .CodeMirror {
        height: auto;
        min-height: 200px;
        background-color: black;
        color: white;
    }
</style>

<!-- Include CodeMirror -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.css" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/codemirror.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.52.2/mode/python/python.min.js"></script>


This is a comprehensive AI fairness exploration framework. 
Visit the <a href="quickstart/" markdown="span">quickstart</a> for a tour.
You may also watch the introductory tutorial, read the full documentation and recipes, or try
the library in your browser.<br>

<iframe width="280" height="157" src="https://www.youtube.com/embed/vJIK3Kc65pA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>
<br>
<br>
<h1 style="margin-bottom: 0px;">Try it here</h1>
Run FairBench from your browser in console integration mode (fallbacks to ascii visualization). Read more in the documentation.
Edit the snippet to be executed and run it below. The minimal library installation only contains textual visualization.

<textarea class="code-block" id="code" rows="40">sensitive = ["M","F","M","F","M","F","M"]
y = [1, 1, 0, 0, 1, 0, 1]
yhat = [1, 1, 1, 0, 0, 0, 0]
scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

report = fb.multireport(
    predictions=yhat, 
    labels=y, 
    scores=scores, 
    sensitive=fb.Fork(fb.categories@ sensitive)
)
fb.describe(report)

fb.text_visualize(report.min.avgrepr.explain)
fb.text_visualize(report.min.avgrepr.explain.explain)</textarea>
<br>
<button id="run" onclick="evaluatePython()"><span class="icon-green">&#9654;</span> Run snippet</button>
<button id="restart" onclick="restartPython()"><span class="icon-blue">&#x1F504;</span> Restart</button>
<a href="https://pyodide.org/en/stable/">Powered by pyodyne</a>
<pre class="code-block" id="output" style="width: 100%; resize: vertical; overflow: auto; max-height: 600px;" rows="30" disabled></pre>

<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script>
    const output = document.getElementById("output");
    const codeTextarea = document.getElementById("code");
    const run = document.getElementById("run");
    const restart = document.getElementById("restart");
    var output_value = "";

    // Initialize CodeMirror on the textarea
    var codeEditor = CodeMirror.fromTextArea(codeTextarea, {
        lineNumbers: true,
        mode: "python",
        theme: "default",
        indentUnit: 4,
        smartIndent: true,
        matchBrackets: true,
        autoCloseBrackets: true
    });

    function convertUndefinedToNone(value) {
        return value === undefined ? "None" : value;
    }

    function ansiToHtml(ansiString) {
        const ansiToHtmlMap = {
            "\u001b[91m": "</span><span style='color:#FF5555'>",       // Red
            "\u001b[92m": "</span><span style='color:#50FA7B'>",       // Green
            "\u001b[93m": "</span><span style='color:#F1FA8C'>",       // Yellow
            "\u001b[94m": "</span><span style='color:#BD93F9'>",       // Blue
            "\u001b[95m": "</span><span style='color:#FF79C6'>",       // Magenta
            "\u001b[96m": "</span><span style='color:#8BE9FD'>",       // Cyan
            "\u001b[93;1m": "</span><span style='color:#FFFF55'>",     // Bright Yellow
            "\u001b[96;1m": "</span><span style='color:#55FFFF'>",     // Bright Cyan
            "\u001b[91;1m": "</span><span style='color:#FFAAAA'>",     // Bright Red
            "\u001b[92;1m": "</span><span style='color:#AAFFAA'>",     // Bright Green
            "\u001b[94;1m": "</span><span style='color:#AAAAFF'>",     // Bright Blue
            "\u001b[95;1m": "</span><span style='color:#FFAAFF'>",     // Bright Magenta
            "\u001b[38;5;208m": "</span><span style='color:#FFA500'>", // Orange
            "\u001b[38;5;202m": "</span><span style='color:#FF4500'>", // Dark Orange
            "\u001b[38;5;198m": "</span><span style='color:#FF69B4'>", // Pink
            "\u001b[38;5;165m": "</span><span style='color:#A020F0'>", // Purple
            "\u001b[38;5;34m": "</span><span style='color:#228B22'>",  // Forest Green
            "\u001b[38;5;70m": "</span><span style='color:#008080'>",  // Teal
            "\u001b[38;5;69m": "</span><span style='color:#00FFFF'>",  // Aqua
            "\u001b[38;5;220m": "</span><span style='color:#FFD700'>", // Gold
            "\u001b[38;5;82m": "</span><span style='color:#32CD32'>",  // Lime Green
            "\u001b[38;5;203m": "</span><span style='color:#FA8072'>", // Salmon
            "\u001b[38;5;166m": "</span><span style='color:#FF7F50'>", // Coral
            "\u001b[38;5;99m": "</span><span style='color:#DA70D6'>",  // Orchid
            "\u001b[38;5;64m": "</span><span style='color:#808000'>",  // Olive Green
            "\u001b[38;5;208;1m": "</span><span style='color:#FFA07A'>", // Bright Orange
            "\u001b[38;5;56m": "</span><span style='color:#9400D3'>",  // Dark Violet
            "\u001b[38;5;123m": "</span><span style='color:#4682B4'>", // Steel Blue
            "\u001b[0m": "</span>"                             // Reset / End
        };
    
        let htmlString = ansiString;
        for (const ansiCode in ansiToHtmlMap) {
            const htmlTag = ansiToHtmlMap[ansiCode];
            htmlString = htmlString.split(ansiCode).join(htmlTag);
        }
    
        return "<span>" + htmlString;
    }

    function addToOutput(s) {
        if (s === undefined) {
        } else {
            output_value += s + "\n";
            const html = ansiToHtml(output_value);
            output.innerHTML = html;
        }
    }

    addToOutput(">>> import fairbench as fb\n");

    async function main() {
        run.disabled = true;
        restart.disabled = true;
        addToOutput("Preparing the browser environment... ");
        let pyodide = await loadPyodide();
        console.log(pyodide.runPython(`
            import sys
            sys.version
        `));
        await pyodide.loadPackage("micropip");
        const micropip = pyodide.pyimport("micropip");
        await micropip.install('fairbench==0.6.0');
        output.value = ">>> import fairbench as fb\n";
        try {
            pyodide.runPython(`import fairbench as fb`);
            addToOutput("done");
        } catch (err) {
            addToOutput( err + "\n");
        }
        run.disabled = false;
        restart.disabled = false;
        return pyodide;
    };
    var pyodideReadyPromise = undefined;
    restart.disabled = true;

    function getCodeString() {
        return codeEditor.getValue();
    }
    
    async function evaluatePython() {
        const command = getCodeString();
        if (pyodideReadyPromise === undefined)
            pyodideReadyPromise = main();
        run.disabled = true;
        restart.disabled = true;
        addToOutput("\n>>> " + command.replace(/\n/g, "\n>>> ") + "\n");

        var logBackup = console.log;

        console.log = function() {
            addToOutput(Array.from(arguments).join(' '));
        };

        let pyodide = await pyodideReadyPromise;
        try {
            let out = pyodide.runPython(command);
            addToOutput(out);
        } catch (err) {
            addToOutput(err);
        }
        console.log = logBackup;
        run.disabled = false;
        restart.disabled = false;
    }
    
    function removeAllCanvas() {
        const elements = document.querySelectorAll('[id^="matplotlib_"]');
        elements.forEach(element => element.remove());
    }

    async function restartPython() {
        output_value = "";
        removeAllCanvas();
        run.disabled = true;
        restart.disabled = true;
        pyodideReadyPromise = undefined;
        pyodideReadyPromise = main();
    }

    // Optional: Run code on Shift+Enter
    document.addEventListener("keydown", function(event) {
        if (event.shiftKey && event.key === "Enter") {
            evaluatePython();
            event.preventDefault();
        }
    });
</script>
</html>
