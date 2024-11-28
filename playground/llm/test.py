from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-125m")
result = generator("What are we having for dinner?")
print(result)