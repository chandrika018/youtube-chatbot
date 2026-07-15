from langchain_core.documents import Document

from vectorstore.faiss_store import KnowledgeBase


def test_knowledge_base_retriever_returns_matching_document(tmp_path):
    store = KnowledgeBase(persist_dir=str(tmp_path / "kb"))
    store.add_documents(
        [
            Document(page_content="Python is excellent for automation and scripting.", metadata={"source": "document"}),
            Document(page_content="Streamlit helps build simple web apps quickly.", metadata={"source": "document"}),
        ],
        source="document",
    )

    docs = store.as_retriever(k=3).invoke("automation")

    assert docs
    assert any("automation" in doc.page_content.lower() for doc in docs)
