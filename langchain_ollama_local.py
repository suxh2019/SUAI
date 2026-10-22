# Import the LangChain prompt template
from langchain_core.prompts import ChatPromptTemplate

# Import the Ollama chat model integration
from langchain_ollama import ChatOllama

# Import the string output parser
from langchain_core.output_parsers import StrOutputParser

'''
(1) Use LangChain with Ollama to run an LLM locally on a laptop;
(2) The langchain-ollama package provides the ChatOllama class for interacting with local Ollama models.

'''

# --------------------------------------------------
# 1. Load a local language model
# --------------------------------------------------

# Connect to the Llama 3.2 1B model running in Ollama
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0.2
)


# --------------------------------------------------
# 2. Create a prompt template
# --------------------------------------------------

# Define the system instruction and user question
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful machine learning tutor. "
        "Explain concepts clearly with examples."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# --------------------------------------------------
# 3. Create the output parser
# --------------------------------------------------

# Convert the model response into a Python string
parser = StrOutputParser()


# --------------------------------------------------
# 4. Build the LangChain pipeline
# --------------------------------------------------

# Connect prompt, language model, and output parser
chain = prompt | llm | parser


# --------------------------------------------------
# 5. Run the language model
# --------------------------------------------------

# Specify the topic to explain
response = chain.invoke({
    "topic": "reinforcement learning"
})

# Display the answer
print("\nLLM Response:")
print(response)


# --------------------------------------------------
# 6. Ask another question
# --------------------------------------------------

# Reuse the same chain with a different topic
response2 = chain.invoke({
    "topic": "Proximal Policy Optimization (PPO)"
})

# Display the second response
print("\nPPO Explanation:")
print(response2)