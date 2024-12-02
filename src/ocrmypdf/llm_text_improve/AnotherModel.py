class PhoGPT:
    _instance = None  # Static variable to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(*args, **kwargs)
        return cls._instance

    def initialize(self, *args, **kwargs):
        # Initialize model attributes here
        self.name = kwargs.get("name", "DefaultModel")

    def summary(self, content):
        summary_content = ""
        return summary_content

    def correct_grammar(self, content):
        corrected_content = ""
        return corrected_content
