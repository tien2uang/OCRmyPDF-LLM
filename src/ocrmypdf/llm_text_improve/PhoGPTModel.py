import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from enum import Enum

from ocrmypdf.llm_text_improve.BaseModel import BaseModel

PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"


class Language(Enum):
    VIETNAMESE = 'vie'
    ENGLISH = 'eng'

class SingletonMeta(type):
    """A metaclass for creating Singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class PhoGPTModel(BaseModel):
    def __init__(self, model_path="vinai/PhoGPT-4B-Chat", device="cuda",dtype=torch.bfloat16):
        super().__init__(model_path=model_path,device=device)
        self.dtype = dtype
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        """Load the tokenizer."""
        print("Loading tokenizer...")
        return AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _load_model(self):
        """Load the model."""
        print("Loading model...")
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        config.init_device = self.device
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to("cuda")
        model.eval()
        return model

    def generate_response(self, instruction):
        """Generate a response for the given instruction."""

        input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
        input_ids = self.tokenizer(input_prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs=input_ids["input_ids"].to("cuda"),
            attention_mask=input_ids["attention_mask"].to("cuda"),
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            max_new_tokens=1024,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response.split("### Trả lời:")[1]


    


# if __name__ == "__main__":
#     model = PhoGPTModel()
#     content ="Hà Nội là thành phô cổ kính với nhiểu di tích lịch sử. Mỗi năm, hàng triệu du khách đén thăm và chim ngưỡng vẻ đẹp của thành phố này. Nơi đây có những con phồ cổ và những quán cà phê rất nỗi tiếng. Dù thời tiết có thay đổi, nhưng Hà Nội vẫ luôn thu hút du khách từ khắp nơi."
#     corrected_content = model.correct_grammar(content)
#     model = PhoGPTModel()
#     print(model.summarize(corrected_content))
