# CSI Document Search Demo

A small command-line project for exploring semantic document search with CSI VIT Pune examples. You can load a PDF or text file, split it into passages, and ask questions to retrieve the closest matches.

The script uses Sentence Transformers for embeddings and ChromaDB for the vector index. **It does not call a language model to write new answers**: the current output is a selection of relevant passages from the uploaded document.

## Run it

Requires Python and these packages:

```bash
python -m pip install chromadb sentence-transformers PyPDF2 numpy
python rag_chatbot.py
```

At the document prompt, enter the path to a `.pdf` or plain-text file. Press Enter to create and load the included CSI example. Type a question at the next prompt; use `quit` or `exit` to finish.

The first run may download the `sentence-transformers/all-MiniLM-L6-v2` model, so an internet connection can be needed for setup.

## How it works

1. Load text from a file or extract text from PDF pages.
2. Split the content into passages and retain page metadata.
3. Convert passages into embeddings with MiniLM.
4. Store vectors in a local ChromaDB collection named `csi_docs`.
5. Embed each question and retrieve up to five nearby passages.
6. Display passage previews and short excerpts for the user to inspect.

The vector database is written under `chroma_db/`. The current script **recreates the collection when it starts**, so previous indexed content is replaced. Keep the source files if you want to index them again.

## Files

- `rag_chatbot.py` — document loading, chunking, indexing, and interactive search
- `demo_csi.txt` — sample text used for quick testing
- `chroma_db/` — generated local index (ignored by Git; recreated when the script starts)

## Limitations

This is a retrieval demo, not a verified question-answering assistant. The similarity scores are search signals, not factual confidence estimates. PDF extraction works best on files that contain selectable text; image-only scans may return little or no content. Always check the displayed source passages before relying on an answer.
