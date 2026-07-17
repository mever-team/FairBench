import fairbench as fb

# model = fb.bench.text.Transformer("facebook/opt-125m")
model = fb.bench.text.Ollama("llama3.2:latest")

print(model("Hi!"))
