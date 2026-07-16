from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.chat_agent import ChatAgent
from loaders.youtube import extract_video_id, load_youtube_transcript
from loaders.documents import load_documents_from_paths
from langchain_core.documents import Document
from retrievers.rag_retriever import CombinedRetriever
from vectorstore.faiss_store import KnowledgeBase


def test_extract_video_id_valid_url():
    assert extract_video_id('https://www.youtube.com/watch?v=dQw4w9WgXcQ') == 'dQw4w9WgXcQ'


def test_extract_video_id_invalid_url():
    assert extract_video_id('not-a-url') is None


def test_load_document_rejects_unsupported_format(tmp_path):
    bad_file = tmp_path / 'notes.unsupported'
    bad_file.write_text('hello')
    try:
        load_documents_from_paths([str(bad_file)])
    except ValueError as exc:
        assert 'Unsupported' in str(exc)
    else:
        raise AssertionError('Expected ValueError for unsupported file type')


def test_load_youtube_transcript_falls_back_when_fetch_fails(monkeypatch):
    class DummyTranscriptApi:
        @staticmethod
        def fetch(*args, **kwargs):
            raise RuntimeError('blocked by YouTube')

    monkeypatch.setattr('loaders.youtube.YouTubeTranscriptApi', DummyTranscriptApi)
    monkeypatch.setattr('loaders.youtube._fetch_video_title', lambda video_id: 'Demo title')

    payload = load_youtube_transcript('https://www.youtube.com/watch?v=dQw4w9WgXcQ')

    assert payload['video_id'] == 'dQw4w9WgXcQ'
    assert 'Transcript unavailable' in payload['text']
    assert payload['title'] == 'Demo title'


def test_knowledge_base_reset_rebuilds_fresh_store(tmp_path):
    persist_dir = tmp_path / 'kb'
    store = KnowledgeBase(persist_dir=str(persist_dir))

    store.add_documents([Document(page_content='old document content', metadata={'source': 'document'})], source='document')
    store.reset()
    store.add_documents([Document(page_content='new document content', metadata={'source': 'document'})], source='document')

    retrieved = store.as_retriever(k=3).invoke('new document content')

    assert any('new document content' in doc.page_content for doc in retrieved)
    assert not any('old document content' in doc.page_content for doc in retrieved)


def test_chat_agent_can_invoke_without_checkpointer_config():
    agent = ChatAgent()
    result = agent.build_graph().invoke({
        'question': 'What is this about?',
        'context': '',
        'answer': '',
        'source': 'both',
        'sources': [],
    })

    assert 'answer' in result
    assert isinstance(result['answer'], str)


def test_combined_retriever_prioritizes_relevant_documents():
    class DummyEmbeddingModel:
        # Mocking embeddings vector search parameters
        def embed_query(self, query):
            return [1.0 if "python" in query.lower() else 0.0] * 384
            
        def embed_documents(self, texts):
            return [[1.0 if "python" in text.lower() else 0.0] * 384 for text in texts]

    class DummyRetriever:
        def __init__(self, docs):
            self.docs = docs

        def invoke(self, question):
            return self.docs

    class DummyStore:
        def __init__(self, docs):
            self.docs = docs
            # Mock model linkage ensures accurate evaluation
            self.embedding_model = DummyEmbeddingModel()

        def as_retriever(self, k=4, **kwargs):
            return DummyRetriever(self.docs[:k])

    youtube_docs = [Document(page_content='The weather is sunny and calm today.', metadata={'source': 'youtube'})]
    document_docs = [Document(page_content='Python programming basics for beginners.', metadata={'source': 'document'})]

    retriever = CombinedRetriever(
        youtube_store=DummyStore(youtube_docs),
        document_store=DummyStore(document_docs),
    )

    result = retriever.retrieve('Tell me about Python programming', source='both')

    assert result[0].page_content.startswith('Python programming')


def test_hybrid_retriever_falls_back_to_keyword_matching(tmp_path):
    store = KnowledgeBase(persist_dir=str(tmp_path / 'hybrid-kb'))
    store._documents = [
        Document(page_content='Python programming basics for beginners.', metadata={'source': 'document', 'title': 'Python Basics'})
    ]

    retriever = store.as_retriever(k=2)
    result = retriever.invoke('beginner python')

    assert any('Python programming' in doc.page_content for doc in result)