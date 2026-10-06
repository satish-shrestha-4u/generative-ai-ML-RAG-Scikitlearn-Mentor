import streamlit as st

from llama_index.core.chat_engine.types import BaseChatEngine

from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ML Mentor for Scikit-Learn",
    page_icon="🤖",
    layout="wide",
)


# ---------------------------------------------------------
# INITIALISE CHAT ENGINE
# ---------------------------------------------------------

def initialise_chat_engine() -> BaseChatEngine:
    """
    Initialise the LLM, embedding model and RAG chat engine.
    """

    llm = initialise_llm()
    embed_model = get_embedding_model()

    chat_engine = get_chat_engine(
        llm=llm,
        embed_model=embed_model,
    )

    return chat_engine


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    with st.spinner("Loading ML Mentor..."):
        st.session_state.chat_engine = initialise_chat_engine()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.title("🧠 ML Mentor")

    st.markdown(
        """
        ### Your Personal ML Expert

        Ask questions about:

        - Python Machine Learning
        - Scikit-Learn
        - Preprocessing
        - Pipelines
        - Cross-validation
        - GridSearchCV
        - Model evaluation
        - Classification
        - Regression
        - K-Means
        - PCA
        - DBSCAN
        - and more
        """
    )

    st.divider()

    st.subheader("Knowledge Base")

    st.write(
        "ML Mentor answers using the information "
        "retrieved from the local knowledge base."
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):

        st.session_state.messages = []

        # Re-create the chat engine so its internal memory is also reset.
        with st.spinner("Resetting ML Mentor..."):
            st.session_state.chat_engine = initialise_chat_engine()

        st.rerun()


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.title("🧠 ML Mentor for Scikit-Learn")

st.markdown(
    """
    **Your Personal Machine Learning & Scikit-Learn Expert**

    Ask questions, request explanations, work through examples,
    or test your knowledge with exam-style questions.
    """
)

st.divider()


# ---------------------------------------------------------
# WELCOME MESSAGE
# ---------------------------------------------------------

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Ask me anything about Machine Learning "
        "and Scikit-Learn."
    )

    st.markdown("### Try asking:")

    example_questions = [
        "What is the difference between KFold and StratifiedKFold?",
        "Explain GridSearchCV with an example.",
        "Why should preprocessing be inside a Pipeline?",
        "What does n_init mean in K-Means?",
        "Give me a Scikit-Learn exam question about nested cross-validation.",
    ]

    for question in example_questions:
        st.markdown(f"- {question}")


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display retrieved sources if available
        if message.get("sources"):

            with st.expander("📚 Retrieved sources"):

                for source in message["sources"]:

                    st.markdown(source)


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

user_question = st.chat_input(
    "Ask ML Mentor a question..."
)


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

if user_question:

    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(user_question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = st.session_state.chat_engine.chat(
                    user_question
                )

                answer = str(response)

                # ---------------------------------------------
                # Retrieve source information
                # ---------------------------------------------

                sources = []

                source_nodes = getattr(
                    response,
                    "source_nodes",
                    []
                )

                for node in source_nodes:

                    metadata = getattr(
                        node,
                        "metadata",
                        {}
                    )

                    file_name = (
                        metadata.get("file_name")
                        or metadata.get("file_path")
                        or "Unknown source"
                    )

                    score = getattr(
                        node,
                        "score",
                        None
                    )

                    if score is not None:

                        sources.append(
                            f"- `{file_name}` "
                            f"(similarity: {score:.3f})"
                        )

                    else:

                        sources.append(
                            f"- `{file_name}`"
                        )

                # ---------------------------------------------
                # Display answer
                # ---------------------------------------------

                st.markdown(answer)

                # ---------------------------------------------
                # Display sources
                # ---------------------------------------------

                if sources:

                    with st.expander("📚 Retrieved sources"):

                        for source in sources:
                            st.markdown(source)

                # ---------------------------------------------
                # Save assistant message
                # ---------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                error_message = (
                    "Sorry, something went wrong while "
                    f"processing your question.\n\n"
                    f"`{e}`"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )
