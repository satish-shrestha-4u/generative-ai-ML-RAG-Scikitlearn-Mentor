# 🤖 ML Mentor for Scikit-Learn

> A Generative AI Retrieval-Augmented Generation (RAG) chatbot designed to help users learn Machine Learning, Python, and Scikit-Learn through contextual explanations, examples, quizzes, and exam-style practice.

## 🔗 <a href="https://generative-ai-ml-rag-scikitlearn-mentor.streamlit.app/" target="_blank">Live App</a>

---

## 📌 Project Overview

**ML Mentor for Scikit-Learn** is an AI-powered learning assistant built using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on the general knowledge of a Large Language Model (LLM), the application retrieves relevant information from a dedicated Machine Learning knowledge base and provides that context to the LLM before generating a response.

This approach helps the application provide responses that are more relevant to the learning material and better aligned with **Python, Machine Learning, and Scikit-Learn** concepts.

The application can be used as an interactive:

- 📚 Learning assistant
- 💬 Machine Learning chatbot
- 🧠 Concept explainer
- 📝 Quiz generator
- 🎯 Exam-practice assistant
- 🔎 Knowledge-base question-answering system

---

## 🧠 How the RAG System Works

The application follows a Retrieval-Augmented Generation pipeline:

```text
User Question
      ↓
Query Processing
      ↓
Vector Search
      ↓
Relevant Knowledge Retrieved
      ↓
Context + User Question
      ↓
Large Language Model
      ↓
Generated Response
      ↓
User
```

### 1. User asks a question

For example:

> "What is the difference between Random Forest and Gradient Boosting?"

### 2. Retrieve relevant knowledge

The application searches the Scikit-Learn / Machine Learning knowledge base using vector similarity.

### 3. Build contextual prompt

The retrieved information is combined with the user's question.

### 4. Generate the answer

The contextual information is passed to the LLM, which generates the final response.

This allows the application to use **external knowledge retrieval + LLM generation** rather than depending entirely on the model's pretrained knowledge.

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   RAG Engine        │
                    │     LlamaIndex      │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Vector Search   │   │   Groq LLM      │
          │ + Embeddings    │   │ Response Gen.   │
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   ▼                     │
          ┌─────────────────┐            │
          │ ML Knowledge    │            │
          │ Base            │────────────┘
          └─────────────────┘
                              
                               ▼
                    ┌─────────────────────┐
                    │   Final Answer      │
                    └─────────────────────┘
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Streamlit** | Interactive web application |
| **LlamaIndex** | RAG orchestration and retrieval |
| **Hugging Face** | Text embedding models |
| **Groq** | Large Language Model inference |
| **Vector Search** | Semantic retrieval from the knowledge base |
| **Scikit-Learn** | Machine Learning concepts covered by the assistant |
| **Git / GitHub** | Version control and project management |

---

## 📁 Main Components

### `streamlit_app.py`

The main application interface.

It is responsible for:

- Creating the Streamlit UI
- Receiving user questions
- Displaying generated responses
- Connecting the interface with the RAG engine

### `src/engine.py`

Contains the core RAG logic.

It manages the process of:

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
LLM
   ↓
Response
```

### `src/model_loader.py`

Responsible for loading and configuring the models used by the application, including the embedding model and LLM components.

### `src/config.py`

Centralises application configuration and environment-dependent settings.

### `data/ml_mentor_knowledge_base/`

Contains the domain-specific knowledge used by the retrieval system.

The knowledge base focuses on topics related to:

- Python
- Machine Learning
- Scikit-Learn
- Model evaluation
- Machine Learning concepts
- Practical examples

---

## 🔍 Why RAG?

A normal LLM chatbot might answer a Machine Learning question using only information contained in its pretrained model.

RAG introduces an additional step:

```text
LLM only
Question → LLM → Answer

RAG
Question → Retrieve Knowledge → LLM + Context → Answer
```

This makes it possible to build an assistant around a **specific knowledge base** rather than relying exclusively on general model knowledge.

For an educational application, this is particularly useful because the assistant can retrieve information relevant to the project's learning material before generating an explanation.

---

## 🎯 Key Features

### 💬 Interactive Q&A

Users can ask natural-language questions about Machine Learning and Scikit-Learn.

### 📚 Context-Aware Answers

The RAG pipeline retrieves relevant information from the project's knowledge base before generating a response.

### 🧠 Machine Learning Mentor

The application focuses on Machine Learning concepts and Scikit-Learn-related topics.

### 📝 Learning Support

The assistant can help users understand concepts through explanations and examples.

### 🎯 Exam Practice

The application can be used for quiz-style and exam-style Machine Learning practice.

### ⚡ Fast LLM Inference

Groq is used for fast Large Language Model inference.

---

## 📈 Possible Future Improvements

Future versions could include:

- 📊 RAG evaluation using dedicated evaluation metrics
- 🧪 Automated question-answer evaluation
- 📚 More comprehensive Machine Learning documentation
- 🧠 More conversation memory
- 🎯 Personalized learning paths
- 📝 Automatic quiz scoring
- 📈 User progress tracking
- 🔎 Improved retrieval and reranking
- 💡 Source citations for retrieved information
- 🛡️ Improved hallucination detection
- ⚡ Retrieval and response caching
- 🧪 Automated testing for the RAG pipeline

---

## 👨‍💻 Author

**Satish Shrestha**

<a href="https://github.com/satish-shrestha-4u" target="_blank">GitHub</a>

<a href="https://www.linkedin.com/in/shrestha-satish/" target="_blank">LinkedIn</a>

---

## ⭐ Project

If you find this project useful, feel free to explore the repository, experiment with the RAG pipeline, and build your own domain-specific AI assistant.

**Built with Python, RAG, LlamaIndex, Hugging Face, Groq, and Streamlit. 🤖**
