"""Local retrieval-augmented generation (RAG) using LangChain and Ollama.

Install:
    python -m pip install -U langchain-core langchain-ollama langchain-text-splitters
    ollama pull llama3.2:1b
    ollama pull nomic-embed-text
Run (Ollama must be running):
    python langchain_rag_ollama_cpu.py
Optional: put .txt files in a folder called 'rag_docs' next to this script.

llama3.2:1b generates answers, while nomic-embed-text creates vector embeddings used for document retrieval.

The main lesson is that RAG retrieves information at inference time without retraining the LLM. This makes it
 particularly useful for building an assistant that answers questions about a changing collection of research 
 papers or technical documents.
"""

from pathlib import Path  # Work with files in a platform-independent way.

from langchain_core.documents import Document  # Wrap text and its source metadata.
from langchain_core.prompts import ChatPromptTemplate  # Build reusable LLM prompts.
from langchain_core.output_parsers import StrOutputParser  # Convert AI messages to strings.
from langchain_ollama import ChatOllama, OllamaEmbeddings  # Locally hosted generation and embeddings.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Divide lengthy text into chunks.

# These are model names installed locally through `ollama pull`.
CHAT_MODEL = "llama3.2:1b"  # Small text-generation model for ordinary laptops.
EMBED_MODEL = "nomic-embed-text"  # Text-embedding model for semantic retrieval.
TOP_K = 3  # Retrieve at most three relevant chunks per question.

# Find local text files relative to the script, not the user's shell directory.
DOCS_FOLDER = Path(__file__).resolve().parent / "rag_docs"

# Provide example knowledge if the user hasn't created any document files yet.
SAMPLE_TEXTS = {
    "ppo.txt": (
        "Proximal Policy Optimization (PPO) is an on-policy reinforcement learning "
        "algorithm. It collects rollouts from its current policy and uses a clipped "
        "surrogate objective to limit large policy updates. PPO often uses a value "
        "network and generalized advantage estimation (GAE)."
    ),
    "sac.txt": (
        "Soft Actor-Critic (SAC) is an off-policy reinforcement learning "
        "algorithm for continuous control. It uses a replay buffer, twin Q-networks, "
        "a stochastic policy, and entropy regularization to encourage exploration."
    ),
    "td3.txt": (
        "Twin Delayed DDPG (TD3) is an off-policy continuous-control algorithm. "
        "It uses twin critics to reduce Q-value overestimation, delayed actor "
        "updates, and target policy smoothing with clipped noise."
    ),
    "iql.txt": (
        "Implicit Q-Learning (IQL) is an offline reinforcement learning algorithm. "
        "It learns values with expectile regression and extracts a policy using "
        "advantage-weighted behavioral cloning without evaluating arbitrary "
        "out-of-distribution actions during value updates."
    ),
}


def load_documents():  # Load local .txt files or fall back to demonstration data.
    documents = []  # Accumulate LangChain Document objects.
    if DOCS_FOLDER.exists():  # Check whether a user document folder exists.
        for path in sorted(DOCS_FOLDER.glob("*.txt")):  # Process each text file.
            text = path.read_text(encoding="utf-8").strip()  # Read UTF-8 contents.
            if text:  # Ignore empty files.
                documents.append(Document(page_content=text, metadata={"source": path.name}))
    if not documents:  # Make the demo work without any user-provided files.
        for filename, text in SAMPLE_TEXTS.items():  # Load built-in RL notes.
            documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents  # Return documents with source names.


def make_chunks(documents):  # Split long source documents into smaller passages.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Aim for chunks of about 400 characters.
        chunk_overlap=60,  # Repeat up to 60 characters for continuity.
    )
    return splitter.split_documents(documents)  # Preserve source metadata in chunks.


def build_index(chunks, embeddings):  # Embed chunks once and keep vectors in memory.
    vectors = embeddings.embed_documents([d.page_content for d in chunks])
    return list(zip(chunks, vectors))  # Pair each source chunk with its vector.


def cosine_similarity(a, b):  # Compute the cosine similarity of two vectors.
    dot = sum(x * y for x, y in zip(a, b))  # Compute the inner product.
    norm_a = sum(x * x for x in a) ** 0.5  # Length of the first vector.
    norm_b = sum(y * y for y in b) ** 0.5  # Length of the second vector.
    return dot / (norm_a * norm_b + 1e-12)  # Avoid dividing by zero.


def retrieve(question, index, embeddings, k=TOP_K):  # Find relevant passages.
    query_vector = embeddings.embed_query(question)  # Convert question to an embedding.
    scored = [
        (cosine_similarity(query_vector, vector), doc)  # Score each document chunk.
        for doc, vector in index  # Iterate over the local document index.
    ]
    scored.sort(key=lambda item: item[0], reverse=True)  # Best match first.
    return scored[:k]  # Return top matching passages and their similarity scores.


def answer_question(question, index, embeddings, chain):  # Run RAG end to end.
    matches = retrieve(question, index, embeddings)  # Retrieve relevant text.
    context = "\n\n".join(  # Combine retrieved passages with source labels.
        f"[Source: {doc.metadata['source']}]\n{doc.page_content}"
        for _, doc in matches
    )
    answer = chain.invoke({"context": context, "question": question})  # Ask the LLM.
    print("\nANSWER:\n" + answer)  # Display the response.
    print("\nRETRIEVED SOURCES:")  # Report source files for transparency.
    for score, doc in matches:  # Print each retrieved source and match score.
        print(f"  {doc.metadata['source']} (cosine similarity: {score:.3f})")


def main():  # Entry point for the demonstration.
    docs = load_documents()  # Read the available text documents.
    chunks = make_chunks(docs)  # Split documents into retrievable chunks.
    print(f"Loaded {len(docs)} document(s) and {len(chunks)} chunk(s).")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  # Create the local embedder.
    index = build_index(chunks, embeddings)  # Embed source chunks once.
    llm = ChatOllama(model=CHAT_MODEL, temperature=0)  # Create the local chat model.
    prompt = ChatPromptTemplate.from_messages([  # Specify grounded answering rules.
        ("system", "Answer using ONLY the supplied context. If the context does "
         "not contain the answer, say 'I do not know from these documents.' "
         "Cite the source filenames in your answer.\n\nContext:\n{context}"),
        ("human", "{question}"),  # Insert the current user question.
    ])
    chain = prompt | llm | StrOutputParser()  # Prompt -> local LLM -> string.

    question = "How are PPO and SAC different?"  # Start with a sample question.
    answer_question(question, index, embeddings, chain)  # Demonstrate one RAG query.

    while True:  # Continue accepting questions until the user exits.
        question = input("\nAsk about your documents (or 'quit'): ").strip()
        if question.lower() in {"quit", "exit", "q"}:  # End the interactive session.
            break
        if question:  # Ignore empty questions.
            answer_question(question, index, embeddings, chain)  # Retrieve and answer.


if __name__ == "__main__":  # Run only when launched directly as a script.
    main()  # Start the application.
