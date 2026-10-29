"""
L7 — Expanded Semantic Search App  (STARTER)
=============================================
Run with:
    streamlit run starter.py

Your goal: extend a ChromaDB-backed Streamlit search tool with source
filtering, result counts, and expandable full text.

Required features:
    1. Load and chunk .txt / .md files from a docs/ folder
    2. Embed and store chunks in a ChromaDB collection (persistent)
    3. Sidebar multiselect to filter by source filename
    4. Search bar that queries ChromaDB (with optional where filter)
    5. "Showing X of Y total documents" count above results
    6. Each result: title (filename), preview (first 150 chars),
       st.expander showing full chunk text

Bonus (pick at least one):
    - A "Similar to this" button that re-queries using that chunk's text
    - Relevance badge (green/yellow/red) based on distance thresholds
    - Sidebar stat: total unique source files indexed

Key concepts:
    ChromaDB where filter for source files:
        where={"source": {"$in": selected_sources}}   # filter to selection
        # omit `where` entirely when no filter is active

    Reading all files from a folder:
        from pathlib import Path
        docs_dir = Path("docs")
        for file in docs_dir.glob("*.md"):
            text = file.read_text()
            # chunk and add to collection with metadata={"source": file.name}

    Counting total docs in collection:
        collection.count()
"""

import streamlit as st
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer


DOCS_DIR   = Path("docs")
CHUNK_SIZE = 500
OVERLAP    = 100
N_RESULTS  = 5

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔎",
    layout="wide"
)

# ── Model & ChromaDB ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_or_create_collection(
        name="docs_search",
        configuration={"hnsw": {"space": "cosine"}}
    )

    return model, collection 

# ── Index documents ───────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fixed-size chunking with overlap."""
    chunks = []
    step = size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + size]
        if chunk.strip():
            chunks.append(chunk)

        if i + size >= len(text):
            break

    return chunks


model, collection = load_resources()


def index_docs():
    """Read docs/ folder and upsert all chunks into ChromaDB."""
    all_chunks = []
    all_ids = []
    all_metadata = []

    for file in DOCS_DIR.iterdir():
        if file.suffix not in (".txt", ".md"):
            continue

        text = file.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file.name}_chunk_{i}")
            all_metadata.append({"source": file.name})

    if all_chunks:
        embeddings = model.encode(all_chunks).tolist()

        collection.upsert(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadata
        )


if collection.count() == 0:
    index_docs()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Filters")

# Get the list of all unique source filenames from the collection
#       Hint: collection.get(include=["metadatas"])["metadatas"]
# st.sidebar.multiselect to let users pick which sources to search
#       Store selection in `selected_sources`

# Get metadata from all stored chunks
stored_data = collection.get(include=["metadatas"])

# Extract unique source filenames
sources = sorted({
    item["source"]
    for item in stored_data["metadatas"]
    if item and "source" in item
})

# Let users select which files to search
selected_sources = st.sidebar.multiselect(
    "Filter by source",
    options=sources
)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("Semantic Search")

query = st.text_input("Search your documents", placeholder="Type a question or phrase...")

if query:
    # Build the `where` filter dict — only add it when selected_sources
    # is non-empty and not equal to all sources

    where_filter = None

    if selected_sources and len(selected_sources) < len(sources):
        where_filter = {
            "source": {"$in": selected_sources}
        }

    # Query the collection
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=N_RESULTS,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    total = collection.count()
    # Show "Showing X of Y total documents"

    shown = len(results["documents"][0])

    st.write(f"Showing {shown} of {total} total documents")


    # Iterate over results and for each chunk display:
    for i in range(shown):
        doc = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        source = metadata["source"]

        # Bonus: Colored relevance indicator
        if distance < 0.5:
            relevance = "🟢 High"
        elif distance < 1.0:
            relevance = "🟡 Medium"
        else:
            relevance = "🔴 Low"

        st.subheader(f"{source} — {relevance}")
        st.caption(f"Distance: {distance:.3f}")

        # First 150 characters
        st.write(doc[:150] + ("..." if len(doc) > 150 else ""))

        # Expand to view the complete chunk
        with st.expander("Show full text"):
            st.write(doc)

        st.divider()