# FairBench

<style>
    #output {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        padding: 10px;
        font-family: monospace;
    }

    #code {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        font-family: monospace;
        spellcheck: false;
        autocorrect: off;
    }
</style>


This is a comprehensive AI fairness exploration framework. 
Visit one of the links below for a quick introduction, or if you came here to get a sense of report entries.
You may also watch the introductory tutorial, or read the full documentation. You can also try
the library in action at the bottom of this page.

<div style="display: flex; flex-wrap: wrap; gap: 10px;" markdown="span">

  <a href="quickstart/" style="border: 1px solid black; padding: 10px; flex: 1; text-align: center;" markdown="span">
    **Quickstart**
  </a>

  <a href="basics/forks/" style="border: 1px solid black; padding: 10px; flex: 1; text-align: center;" markdown="span">
    <b>Sensitive attribute forks</b>
  </a>

  <a href="record/comparisons/" style="border: 1px solid black; padding: 10px; flex: 1; text-align: center;" markdown="span">
    <b>Metric comparisons</b>
  </a>


<iframe width="280" height="157" src="https://www.youtube.com/embed/vJIK3Kc65pA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

</div>

<br>
<h1 style="margin-bottom: 0px;">Try it here</h1>
Write and run code snippets that use FairBench to perform fairness analysis.
You may start with the pre-filled example for creating fairness reports - follow
some of the links above to explain what resutls mean.

<textarea class="code-block" id="code" style="width: 100%;overflow: hidden;resize: none;" rows="10">sensitive = fb.Fork(fb.categories@["Male", "Female", "Male", "Female", "Nonbin"])
y, yhat = [1, 1, 0, 0, 1], [1, 1, 1, 0, 0]
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report, show=False)  # returned value is printed anyway
fb.visualize(report.accuracy)    # in the browser, default engine is ascii</textarea>


<button id="run" onclick="evaluatePython()"><span class="icon-green">&#9654;</span> Run snippet</button>
<button id="restart" onclick="restartPython()"><span class="icon-blue">&#x1F504;</span> Restart</button>
<a href="https://pyodide.org/en/stable/">Powered by pyodyne</a>
<textarea class="code-block" id="output" style="width: 100%;resize: vertical;" rows="20" disabled></textarea>

    
<script src="https://cdn.jsdelivr.net/pyodide/v0.26.2/full/pyodide.js"></script>
<script>
    const output = document.getElementById("output");
    const code = document.getElementById("code");
    const run = document.getElementById("run");
    const restart = document.getElementById("restart");

    function convertUndefinedToNone(value) {
        return value === undefined ? "None" : value;
    }

    function addToOutput(s) {
        if (s === undefined) {
        } else {
            output.value += s + "\n";
            output.scrollTop = output.scrollHeight;
        }
    }

    output.value = ">>> import fairbench as fb\n";

    async function main() {
        run.disabled = true;
        restart.disabled = true;
        output.value += "Preparing the browser environment (this may take a couple of minutes)... ";
        let pyodide = await loadPyodide();
        console.log(pyodide.runPython(`
            import sys
            sys.version
        `));
        await pyodide.loadPackage("micropip");
        const micropip = pyodide.pyimport("micropip");
        await micropip.install('fairbench');
        output.value = ">>> import fairbench as fb\n";
        try {
            pyodide.runPython(`import fairbench as fb`);
        } catch (err) {
            output.value += err + "\n";
        }
        run.disabled = false;
        restart.disabled = false;
        return pyodide;
    };
    var pyodideReadyPromise = undefined;
    restart.disabled = true;

    function getCodeString() {
        const codeElement = document.getElementById("code");
        const codeString = codeElement.value;
        return codeString;
    }
    
    async function evaluatePython() {
        const command = getCodeString();
        if (pyodideReadyPromise === undefined)
            pyodideReadyPromise = main();
        run.disabled = true;
        restart.disabled = true;
        let pyodide = await pyodideReadyPromise;
        output.value += ">>> " + command.replace("\n", "\n>>> ") + "\n";

        var logBackup = console.log;

        console.log = function() {
            addToOutput(Array.from(arguments).join(' '));
        };

        try {
            let output = pyodide.runPython(command);
            addToOutput(output);
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
        output.value = "Restarting python...\n";
        removeAllCanvas();
        run.disabled = true;
        restart.disabled = true;
        pyodideReadyPromise = undefined;
        await main();
    }

    // Run code on Shift+Enter
    document.getElementById("code-editor").addEventListener("keydown", function(event) {
        if (event.key === "Enter" && event.shiftKey) {
            evaluatePython();
            event.preventDefault();
        }
    });
    
    function autoResize() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
    }
    code.style.height = 'auto';
    code.style.height = code.scrollHeight + 'px';
    code.addEventListener('input', autoResize, false);
</script>


<style>
    .icon-green {
        color: green;
    }
    .icon-blue {
        color: blue;
    }
</style>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>


<script>
    Prism.highlightAll(); // Apply syntax highlighting to code

    // Re-run syntax highlighting on code changes
    document.getElementById("code").addEventListener("input", function() {
        Prism.highlightElement(this);
    });
</script>