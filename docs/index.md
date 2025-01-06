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
    }
    .code-block {
        background-color: black;
        color: white;
        border: 1px solid #555555;
        font-family: monospace;
        spellcheck: false;
        margin-top: 0px;
        font-size: 0.8em;
    }
    .icon-green {
        color: green;
    }
    .icon-blue {
        color: blue;
    }
    .CodeMirror {
        font-size: 0.8em;
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

<h1 style="margin-bottom: 0px;">FairBench</h1>

This is a comprehensive AI fairness exploration framework. 
Visit the <a href="quickstart/" markdown="span">quickstart</a> for a tour and read the 
documentation. Or try
the library in your browser below (with only lightweight features).
<br><br>

<button id="run" onclick="evaluatePython()"><span class="icon-green">&#9654;</span> Run snippet</button>
<button id="restart" onclick="restartPython()"><span class="icon-blue">&#x1F504;</span> Restart</button>
<a href="https://pyodide.org/en/stable/">Powered by pyodyne</a>
<pre class="code-block" id="output" style="width: 100%; resize: vertical; overflow: auto; max-height: 600px;" rows="30" disabled></pre>

<textarea class="code-block" id="code" rows="40">
sensitive = ["M","F","M","F","M","F","M"]
y = [1, 1, 0, 0, 1, 0, 1]
yhat = [1, 1, 1, 0, 0, 0, 0]

report = fb.reports.pairwise(
    predictions=yhat, 
    labels=y, 
    sensitive=fb.Dimensions(fb.categories @ sensitive)
)

report.show(env=fb.export.ConsoleTable(legend=False))
report.maxdiff.show() # dot specializes, could also show everything</textarea>

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
        // Maps for ANSI codes to CSS styles
        const colors = {
            30: "color:#CCCCCC", // Light gray (instead of pure black, so it's visible on black)
            31: "color:#E74C3C", // A rich, warm red
            32: "color:#27AE60", // A vibrant, balanced green
            33: "color:#F1C40F", // A bright but not overpowering yellow
            34: "color:#3498DB", // A bold, medium-blue tone
            35: "color:#9B59B6", // A soft magenta/purple
            36: "color:#1ABC9C", // A cool, pleasing cyan
            37: "color:#ECF0F1",  // A near-white, soft tone
            90: "color:#555555", // Bright Black (Gray)
            91: "color:#FF5555", // Bright Red
            92: "color:#50FA7B", // Bright Green
            93: "color:#F1FA8C", // Bright Yellow
            94: "color:#BD93F9", // Bright Blue
            95: "color:#FF79C6", // Bright Magenta
            96: "color:#8BE9FD", // Bright Cyan
            97: "color:#FFFFFF"  // Bright White
        };
    
        const backgrounds = {
            40: "background-color:#000000", // Bg Black
            41: "background-color:#FF0000", // Bg Red
            42: "background-color:#00FF00", // Bg Green
            43: "background-color:#FFFF00", // Bg Yellow
            44: "background-color:#0000FF", // Bg Blue
            45: "background-color:#FF00FF", // Bg Magenta
            46: "background-color:#00FFFF", // Bg Cyan
            47: "background-color:#FFFFFF", // Bg White
            100: "background-color:#555555",// Bright Bg Black (Gray)
            101: "background-color:#FF5555",// Bright Bg Red
            102: "background-color:#50FA7B",// Bright Bg Green
            103: "background-color:#F1FA8C",// Bright Bg Yellow
            104: "background-color:#BD93F9",// Bright Bg Blue
            105: "background-color:#FF79C6",// Bright Bg Magenta
            106: "background-color:#8BE9FD",// Bright Bg Cyan
            107: "background-color:#FFFFFF" // Bright Bg White
        };
    
        // Additional extended 256-color mode (e.g., 38;5;... for foreground, 48;5;... for background)
        // You provided some mappings for these extended colors:
        // Example: "\u001b[38;5;208m" : color:#FFA500 (Orange)
        // We will define a helper to map these if they appear:
        const extendedColors = {
            208: "#FFA500", // Orange
            202: "#FF4500", // Dark Orange
            198: "#FF69B4", // Pink
            165: "#A020F0", // Purple
            34:  "#228B22", // Forest Green
            70:  "#008080", // Teal
            69:  "#00FFFF", // Aqua
            220: "#FFD700", // Gold
            82:  "#32CD32", // Lime Green
            203: "#FA8072", // Salmon
            166: "#FF7F50", // Coral
            99:  "#DA70D6", // Orchid
            64:  "#808000", // Olive Green
            56:  "#9400D3", // Dark Violet
            123: "#4682B4"  // Steel Blue
        };
        
        // Some bright variants you included that have a format like "\u001b[93;1m"
        // These can be handled by applying both the color and style=bold. 
        // We'll handle the '1' (bold) or any style codes generically below.
    
        // Style attributes
        // We'll track them in a state object and rebuild style string whenever something changes.
        let state = {
            color: null,
            background: null,
            bold: false,
            dim: false,
            italic: false,
            underline: false,
            blink: false,
            inverse: false,
            hidden: false,
            strikethrough: false
        };
    
        // Convert the state into a CSS style string
        function stateToStyleString(s) {
            const styleList = [];
            if (s.color) styleList.push(s.color);
            if (s.background) styleList.push(s.background);
            if (s.bold) styleList.push("font-weight:bold");
            if (s.dim) styleList.push("opacity:0.6");
            if (s.italic) styleList.push("font-style:italic");
            if (s.underline) styleList.push("text-decoration:underline");
            if (s.blink) styleList.push("text-decoration:blink");    // Not widely supported
            if (s.inverse) styleList.push("filter:invert(100%)");    // Rough simulation
            if (s.hidden) styleList.push("visibility:hidden");
            if (s.strikethrough) styleList.push("text-decoration:line-through");
            return styleList.join(";");
        }
    
        // We'll build the final result and update spans as we go
        let result = "";
        let openSpan = false;
    
        function openNewSpan() {
            const style = stateToStyleString(state);
            result += "<span" + (style ? " style='" + style + "'" : "") + ">";
            openSpan = true;
        }
    
        function closeSpanIfOpen() {
            if (openSpan) {
                result += "</span>";
                openSpan = false;
            }
        }
    
        // Reset the state to default
        function resetState() {
            state = {
                color: null,
                background: null,
                bold: false,
                dim: false,
                italic: false,
                underline: false,
                blink: false,
                inverse: false,
                hidden: false,
                strikethrough: false
            };
        }
    
        // Update the state and re-open span
        function applyCodes(codes) {
            let needNewSpan = false;
            for (const code of codes) {
                const c = parseInt(code, 10);
                if (c === 0) {
                    // Reset
                    resetState();
                    closeSpanIfOpen();
                    needNewSpan = true;
                } else if (c === 1) {
                    state.bold = true; needNewSpan = true;
                } else if (c === 2) {
                    state.dim = true; needNewSpan = true;
                } else if (c === 3) {
                    state.italic = true; needNewSpan = true;
                } else if (c === 4) {
                    state.underline = true; needNewSpan = true;
                } else if (c === 5) {
                    state.blink = true; needNewSpan = true;
                } else if (c === 7) {
                    state.inverse = true; needNewSpan = true;
                } else if (c === 8) {
                    state.hidden = true; needNewSpan = true;
                } else if (c === 9) {
                    state.strikethrough = true; needNewSpan = true;
                } else if (c >= 30 && c <= 37) {
                    // Set foreground color
                    state.color = colors[c]; needNewSpan = true;
                } else if (c >= 90 && c <= 97) {
                    // Bright foreground
                    state.color = colors[c]; needNewSpan = true;
                } else if (c >= 40 && c <= 47) {
                    // Background color
                    state.background = backgrounds[c]; needNewSpan = true;
                } else if (c >= 100 && c <= 107) {
                    // Bright background
                    state.background = backgrounds[c]; needNewSpan = true;
                } else if (code.startsWith("38;5;")) {
                    // 256-color foreground: 38;5;x
                    const colorIndex = parseInt(code.split(";")[2], 10);
                    if (extendedColors[colorIndex]) {
                        state.color = "color:" + extendedColors[colorIndex];
                    } else {
                        // Fallback if not in extendedColors, just skip or set a default
                        state.color = "color:#FFFFFF";
                    }
                    needNewSpan = true;
                } else if (code.startsWith("48;5;")) {
                    // 256-color background: 48;5;x
                    const colorIndex = parseInt(code.split(";")[2], 10);
                    if (extendedColors[colorIndex]) {
                        state.background = "background-color:" + extendedColors[colorIndex];
                    } else {
                        // Fallback if not in extendedColors
                        state.background = "background-color:#000000";
                    }
                    needNewSpan = true;
                }
            }
    
            if (needNewSpan) {
                // Close and re-open span with new style
                closeSpanIfOpen();
                openNewSpan();
            }
        }
    
        // Regex to match ANSI escape codes (e.g., "\u001b[31m", "\u001b[38;5;123m", etc.)
        const ansiRegex = /\u001b\[((?:\d|;)+)m/g;
        let lastIndex = 0;
        let match;
    
        // Initially open a span for the default state
        openNewSpan();
    
        while ((match = ansiRegex.exec(ansiString)) !== null) {
            const chunk = ansiString.slice(lastIndex, match.index);
            result += chunk; // Add text before the ANSI code
    
            const codes = match[1].split(";"); // Extract the numeric codes
            applyCodes(codes);
    
            lastIndex = ansiRegex.lastIndex;
        }
    
        // Add any remaining text after the last ANSI code
        result += ansiString.slice(lastIndex);
    
        // Close any open spans
        closeSpanIfOpen();
    
        return result;
    }


    function addToOutput(s) {
        if (s === undefined) {
        } else {
            output_value += s + "\n";
            const html = ansiToHtml(output_value);
            output.innerHTML = html;
            output.scrollTop = output.scrollHeight;
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
        await micropip.install('fairbench==0.7.4');
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
