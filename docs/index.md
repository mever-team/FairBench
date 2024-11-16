# FairBench

<style>
    #output {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        padding: 10px;
        font-family: monospace;
        margin-bottom: 25px;
    }
    #code {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        font-family: monospace;
        spellcheck: false;
        autocorrect: off;
        margin-bottom: 25px;
        margin-top: 0px;
    }
    .icon-green {
        color: green;
    }
    .icon-blue {
        color: blue;
    }
</style>


This is a comprehensive AI fairness exploration framework. 
Visit the <a href="quickstart/" markdown="span">quickstart</a> for a tour.
You may also watch the introductory tutorial, read the full documentation and recipes, or try
the library in action at the bottom of this page.

<iframe width="280" height="157" src="https://www.youtube.com/embed/vJIK3Kc65pA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>
<h1 style="margin-bottom: 0px;">Try it here</h1>
Run FairBench from your browser in console integration mode (fallbacks to ascii visualization). Read more in the documentation.
Edit the snippet to be executed and run it bellow.

<textarea class="code-block" id="code" style="width: 100%;overflow: hidden;resize: none;" rows="10">sensitive = fb.Fork(fb.categories@["Male", "Female", "Male", "Female", "Nonbin"])
y, yhat = [1, 1, 0, 0, 1], [1, 1, 1, 0, 0]
report = fb.multireport(predictions=yhat, labels=y, sensitive=sensitive)
fb.describe(report, show=False)  # returned value is printed anyway
fb.visualize(report.accuracy)</textarea>


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
