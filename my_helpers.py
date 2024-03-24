# import schema for chat messages and ChatOpenAI in order to query chat models GPT-3.5-turbo or GPT-4
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI


def is_oblivious(given_answer):
    oblivious_markers = ["I don't know", "I do not know", "don't have that information",
                         "I do not have that information", "cannot answer that"]
    return any(marker in given_answer for marker in oblivious_markers)


def my_agent(given_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
    messages = [
        SystemMessage(content="You are a funny and helpful assistant."),
        AIMessage(content="I will do my best to find the true answer for you!"),
        HumanMessage(content=given_input),
    ]
    response = chat.invoke(messages)
    return response.content


# Test my function
if __name__ == '__main__':
    test_answers = ["I don't know.", "I do not know.", "I don't have that information...", "I cannot answer that",
                    "Sorry, I don't have that information.", "I really don't have that information", "I now know"]
    for answer in test_answers:
        print(f"Is oblivious: {answer} : " + str(is_oblivious(answer)))
