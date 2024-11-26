class LLM:
    def __init__(
        self, model_name="facebook/opt-2.7b", quantization="8bit", device=None
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if quantization == "8bit":
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant_config
            )
        elif quantization == "4bit":
            from transformers import (
                BitsAndBytesConfig,
            )  # pip install transformers bitsandbytes

            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=quant_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def __call__(self, query):
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        output_tokens = self.model.generate(
            inputs.input_ids, max_length=47, num_return_sequences=1
        )
        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        return response
