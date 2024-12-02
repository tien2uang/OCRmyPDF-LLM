from abc import abstractmethod
from enum import Enum



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





class BaseModel(metaclass=SingletonMeta):
    def __init__(self,model_path, device):
        self.model_path = model_path
        self.device = device
    @abstractmethod
    def _load_tokenizer(self):
        """Load the tokenizer."""

    @abstractmethod
    def _load_model(self):
        """Load the model."""

    @abstractmethod
    def generate_response(self, instruction):
        """Generate a response for the given instruction."""

    def summarize(self, content, languages):
        """Summarize the given content."""
        print("Summarizing...")
        if Language.VIETNAMESE in languages:
            instruction = f"Tóm tắt nội dung văn bản sau. Câu trả lời bắt buộc bắt đầu bằng 'Nội dung này nói về':\n{content}"
        elif Language.ENGLISH in languages:
            instruction = f"Summarize the content:\n{content}"
        else:
            instruction = f"Tóm tắt nội dung văn bản sau. Câu trả lời bắt buộc bắt đầu bằng 'Nội dung này nói về':\n{content}"
        return self.generate_response(instruction)

    def correct_grammar(self, content, languages):
        """Correct grammar of the given content."""
        print("Correcting grammar...")
        if Language.VIETNAMESE in languages:
            instruction = f"Sửa lỗi chính tả:\n{content}"
        elif Language.ENGLISH in languages:
            instruction = f"Correct grammar for the content:\n{content}"
        else:
            instruction = f"Sửa lỗi chính tả:\n{content}"
        return self.generate_response(instruction)
