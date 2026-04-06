import ollama
from langchain_ollama import ChatOllama
# Call the 3rd model from your list
# response = ollama.chat(model='gemma3:4b', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])

# # Print the response
# print(response['message']['content'])


# llama3.2:latest
#
from langchain_ollama import ChatOllama

from langchain_ollama import ChatOllama

model = ChatOllama(
    model='llama3.2:latest',
    temperature=0.7
)

# response = model.invoke("Why is the sky blue?")

for chunk in model.stream("Explain why the sky is blue in simple words"):
    print(chunk.content, end="", flush=True)