from langchain_community.document_loaders import ArxivLoader

# Supports all arguments of `ArxivAPIWrapper`
loader = ArxivLoader(
    query="reasoning",
    load_max_docs=2,
    # doc_content_chars_max=1000,
    # load_all_available_meta=False,
    # ...
)

docs = loader.load()
print(docs[0].metadata)