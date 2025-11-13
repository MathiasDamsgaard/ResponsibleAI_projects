from models.OpenChat_3_5 import get_response as get_response_openchat
from models.OpenChat_3_5 import init_model as init_model_openchat



class ChatModel:
    def __init__(self, model_name="openchat"):
        self.model_name = model_name
        if model_name == "openchat":
            self.model = init_model_openchat()
            self.get_response_internal = get_response_openchat
        else:
            raise ValueError("Invalid model name")

    def get_response(self, prompt):
        return self.get_response_internal(prompt, self.model)
