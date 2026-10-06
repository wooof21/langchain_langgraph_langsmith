
from langchain.chat_models import init_chat_model


def create_model(model: str, temperature: float | None = None,  **kwargs):
    config = {
        "model": model,
        "streaming": True,
        "max_retries": 3,
        **kwargs,
    }

    if temperature is not None:
        config["temperature"] = temperature

    return init_chat_model(**config)


class ChatModelFactory:

    @staticmethod
    def openai_4o_mini(temperature=0.7, **kwargs):
        return create_model("gpt-4o-mini", temperature=temperature, **kwargs)

    @staticmethod
    def openai_4o(temperature=0.7, **kwargs):
        return create_model("gpt-4o", temperature=temperature, **kwargs)

    '''
    gpt-6-luna, cannot set temperature to arbitrary values such as 0, 0.2, or 0.7
    only the default value 1 is supported
    '''
    @staticmethod
    def openai_6_luna(**kwargs):
        return create_model("gpt-6-luna", temperature=None, **kwargs)
    
    @staticmethod
    def anthropic(temperature=0.7, **kwargs):
        return create_model(
            "claude-sonnet-4-5-20250929",
            temperature=temperature,
            model_provider="anthropic",
            **kwargs
        )