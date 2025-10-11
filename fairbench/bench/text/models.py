def Transformer(name):
    from transformers import pipeline

    generator = pipeline("text-generation", model=name, max_length=47)
    return lambda prompt: generator("Question: " + prompt + "\nAnswer:")[0][
        "generated_text"
    ]


def Ollama(name):
    import requests

    def run(prompt):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": name,
                "prompt": prompt,
                "stream": False,  # <-- disable streaming explicitly
            },
        )
        data = response.json()
        return data.get("response", "").strip()

    return run
