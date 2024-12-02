
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"


class PhoGPTModel:
    _model_instance = None
    _tokenizer_instance = None

    @staticmethod
    def get_instance(*args, **kwargs):
        # Ensure the model is instantiated only once
        if PhoGPTModel._model_instance is None:
            PhoGPTModel._model_instance = PhoGPTModel(*args, **kwargs)
        return PhoGPTModel._model_instance

    @staticmethod
    def get_tokenizer_instance(*args, **kwargs):
        # Ensure the tokenizer is instantiated only once
        if PhoGPTModel._tokenizer_instance is None:
            PhoGPTModel._tokenizer_instance = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat",
                                                                            trust_remote_code=True)
        return PhoGPTModel._tokenizer_instance

    def __init__(self):
        if PhoGPTModel._model_instance is not None:
            raise Exception("This class is a singleton!")
        model_path = "vinai/PhoGPT-4B-Chat"

        # Load model configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.init_device = "cuda"

        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        # Move model to the correct device
        model = model.to("cuda")

        PhoGPTModel._model_instance = model  # Assign the model instance

    def generate_response(self, instruction, _tokenizer_instance=None):
        input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})

        # Tokenize the input
        input_ids = self.get_tokenizer_instance()(input_prompt, return_tensors="pt")

        # Generate the model's output
        outputs = self.get_instance().generate(
            input_ids=input_ids["input_ids"].to("cuda"),
            attention_mask=input_ids["attention_mask"].to("cuda"),
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            max_new_tokens=1024,
            eos_token_id=self.get_tokenizer_instance().eos_token_id,
            pad_token_id=self.get_tokenizer_instance().pad_token_id
        )

        # Decode the output
        response = self.get_tokenizer_instance().batch_decode(outputs, skip_special_tokens=True)[0]
        response = response.split("### Trả lời:")[1]
        return response

    def summary(self, content):
        instruction = "Tóm tắt đoạn văn:\n" + content
        return self.generate_response(instruction)

    def correct_grammar(self, content):
        instruction = "Sửa chính tả:\n" + content
        return self.generate_response(instruction)
