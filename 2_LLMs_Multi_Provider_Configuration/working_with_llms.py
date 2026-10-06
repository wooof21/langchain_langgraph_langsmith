"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from config.chat_models import ChatModelFactory


"""
LangChain Messages doc: https://docs.langchain.com/oss/python/langchain/messages#messages
"""

load_dotenv()


def demo_init_chat_model():
    model = ChatModelFactory.openai_4o_mini()

    response = model.invoke("What is the capital of France? Answer in one word.")
    print(f"Response from OpenAI: {response.content}")

    # easy to switch model providers
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = ChatModelFactory.anthropic()

        response = claude.invoke("What is the capital of France? Answer in one word.")
        print(f"Response from Anthropic: {response.content}")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": ChatModelFactory.openai_4o_mini(),
        "gpt-4o": ChatModelFactory.openai_4o(),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = ChatModelFactory.anthropic()

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")

"""
Multi turn conversation, appending response and new messages to messages list
remember the converstaion state.
"""
def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    # print("Using message objects:")
    # print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)  # add model's response to the conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")


def exercise():
    """
        EXERCISE: Create a function that:
        1. Takes a question and a list of model names
        2. Gets responses from all models
        3. Returns a dict of {model_name: response}
    
        Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o"]
        """
    prompt = ChatPromptTemplate.from_template("You are a professional instructor. " \
    "Answer the question: {question} in an easy to understand way.")
    
    models = {
        "gpt-4o-mini": ChatModelFactory.openai_4o_mini(),
        "gpt-4o": ChatModelFactory.openai_4o(),
    }

    responses = {}

    for model_name, model in models.items():
        chain = prompt | model
        response = chain.invoke({"question": "What is AI?"})
        responses[model_name] = response.content

    print(responses)


    




if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    # demo_message()
    exercise()
