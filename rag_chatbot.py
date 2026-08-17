# ============================================================
# CSI RAG CHATBOT — FINAL WORKING VERSION
# Compatible with ChromaDB 0.5+ (latest API)
# ============================================================

import os
import sys

# ============================================================
# STEP 1: CHECK DEPENDENCIES
# ============================================================

def check_install(pkg, imp=None):
    if imp is None:
        imp = pkg
    try:
        __import__(imp)
        return True
    except ImportError:
        return False

required = {
    "chromadb": "chromadb",
    "sentence-transformers": "sentence_transformers",
    "PyPDF2": "PyPDF2",
    "numpy": "numpy",
}

print("=" * 60)
print("CHECKING DEPENDENCIES")
print("=" * 60)

missing = []
for pkg, imp in required.items():
    if check_install(pkg, imp):
        print(f"  ✅ {pkg}")
    else:
        print(f"  ❌ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n❌ Missing: {', '.join(missing)}")
    print("\n🔧 Run: pip install chromadb sentence-transformers PyPDF2 numpy")
    sys.exit(1)

print("\n✅ All dependencies found!")

# ============================================================
# STEP 2: IMPORTS
# ============================================================

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import PyPDF2

print("✅ All imports successful!")

# ============================================================
# STEP 3: DOCUMENT LOADER
# ============================================================

def load_pdf(file_path):
    """Load PDF using PyPDF2"""
    pages = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({'content': text, 'page': i+1})
    print(f"📄 Loaded PDF: {len(pages)} pages")
    return pages

def load_text(file_path):
    """Load text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"📝 Loaded text: {len(content)} chars")
    return [{'content': content, 'page': 1}]

# ============================================================
# STEP 4: TEXT SPLITTER
# ============================================================

def split_text(docs, chunk_size=500, overlap=100):
    """Split documents into chunks"""
    chunks = []
    for doc in docs:
        text = doc['content']
        page = doc.get('page', 1)

        # Split by sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        current = ""
        for sent in sentences:
            if len(current) + len(sent) < chunk_size:
                current += sent + ". "
            else:
                if current:
                    chunks.append({'text': current.strip(), 'page': page})
                # Keep overlap
                words = current.split()
                overlap_text = " ".join(words[-10:]) if len(words) > 10 else ""
                current = overlap_text + " " + sent + ". "

        if current:
            chunks.append({'text': current.strip(), 'page': page})

    print(f"✂️  Created {len(chunks)} chunks")
    return chunks

# ============================================================
# STEP 5: VECTOR STORE (NEW CHROMADB API)
# ============================================================

class SimpleRAG:
    def __init__(self):
        print("\n🔢 Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Model loaded!")

        # NEW CHROMADB API: Use PersistentClient instead of Client with Settings
        db_path = "./chroma_db"
        os.makedirs(db_path, exist_ok=True)

        print("📂 Setting up vector database...")
        self.client = chromadb.PersistentClient(path=db_path)

        # Delete old collection if exists
        try:
            self.client.delete_collection("csi_docs")
            print("🗑️  Cleared old collection")
        except Exception:
            pass

        # Create new collection
        self.collection = self.client.create_collection(
            name="csi_docs",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Vector database ready!")

        self.chunks = []

    def ingest(self, file_path):
        """Load and process document"""
        print(f"\n{'='*60}")
        print("📥 INGESTING DOCUMENT")
        print(f"{'='*60}")

        # Load
        if file_path.endswith('.pdf'):
            docs = load_pdf(file_path)
        else:
            docs = load_text(file_path)

        # Split
        self.chunks = split_text(docs)

        # Embed and store
        print("\n🔢 Creating embeddings...")
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        metadatas = [{'page': c['page']} for c in self.chunks]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        print(f"\n✅ Document ready! {len(self.chunks)} chunks stored.")
        return True

    def ask(self, question):
        """Answer question using retrieved chunks"""
        print(f"\n{'='*60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*60}")

        # Embed question
        q_embedding = self.embedder.encode([question])

        # Search
        print("\n🔍 Searching...")
        results = self.collection.query(
            query_embeddings=q_embedding.tolist(),
            n_results=5
        )

        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]

        print(f"\n📚 Found {len(docs)} relevant passages:")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            preview = doc[:150].replace('\n', ' ')
            score = (1 - dist) * 100
            print(f"\n  [{i}] Page {meta['page']} (Relevance: {score:.1f}%)")
            print(f"      {preview}...")

        # Show answer
        print(f"\n{'='*60}")
        print("💡 ANSWER (based on retrieved passages):")
        print(f"{'='*60}")

        answer = "\n\n".join([f"[{i+1}] {d[:300]}..." for i, d in enumerate(docs[:3])])
        print(answer)

        print(f"\n{'='*60}")
        print("💡 TIP: Read the passages above to find your answer.")
        print(f"{'='*60}")

# ============================================================
# STEP 6: MAIN
# ============================================================

def create_demo():
    """Create a demo CSI document"""
    text = """Computer Society of India (CSI) is the first and largest body of computer professionals in India. It was started on 6 March 1965 by a few computer professionals.

CSI VIT Pune is the student branch of CSI at Vishwakarma Institute of Technology, Pune. It was established to promote technical learning among students through workshops, hackathons, seminars, and hands-on projects.

The AI/ML domain at CSI VIT Pune focuses on artificial intelligence and machine learning technologies. Members work on projects involving LLMs, computer vision, NLP, and RAG systems. The domain conducts workshops on transformer models, decision trees, and neural networks.

RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by retrieving relevant documents and providing them as context. It was introduced by Lewis et al. in 2020. RAG reduces hallucinations by grounding answers in retrieved facts.

The Web Development domain at CSI creates and maintains the club's website and web applications. They use React, Node.js, and modern frameworks. They build responsive navigation bars, contact forms, and event explorers.

The App Development domain builds mobile applications using Flutter and Dart. They focus on creating user-friendly interfaces and cross-platform apps.

CSI organizes multiple events throughout the year including hackathons, technical workshops, guest lectures by industry professionals, and coding competitions.

The DSA domain focuses on data structures and algorithms, preparing students for coding interviews and competitive programming contests. They conduct regular practice sessions and mock interviews.

The Blockchain domain explores decentralized applications, smart contracts, and blockchain technology solutions. They work on Ethereum-based projects and study case studies of popular DApps.

CSI VIT Pune has a strong social media presence on Instagram and LinkedIn. The Social Media domain manages content creation, event promotion, and community engagement."""

    with open("demo_csi.txt", 'w', encoding='utf-8') as f:
        f.write(text)
    return "demo_csi.txt"

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           CSI VIT PUNE — RAG CHATBOT                    ║
    ║     Retrieval-Augmented Generation Document Q&A         ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    rag = SimpleRAG()

    # Get file
    file_path = input("📁 Enter document path (or press Enter for demo): ").strip().strip('"')

    if not file_path or not os.path.exists(file_path):
        print("\n💡 Creating demo document...")
        file_path = create_demo()
        print(f"✅ Created: {file_path}")

    # Ingest
    rag.ingest(file_path)

    # Chat
    print("\n" + "="*60)
    print("💬 CHAT STARTED (type 'quit' to exit)")
    print("="*60)

    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if not q:
            continue
        rag.ask(q)