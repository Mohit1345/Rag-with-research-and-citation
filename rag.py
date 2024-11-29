from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import getpass
import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import ArxivLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""
    

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def chunking(pdf_data):
    print("chunking")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = SemanticChunker(embeddings)
    # print("pdf data is ", pdf_data)
    docs = text_splitter.create_documents([pdf_data])
    # print("chunks is , ",docs)
    return docs


import chromadb.api


def arxiv(topic):
    print("Retrieving research...")
    loader = ArxivLoader(
        query=topic,
        load_max_docs=2,
        load_all_available_meta=True,
    )
    docs = loader.load()
    print("docs ", docs)

    ref_json = []
    result = []
    for doc in docs:
        print("doc we got is ", doc)
        temp_json = {
            'Published': doc.metadata['Published'], 
            'Title': doc.metadata['Title'],
            'Authors': doc.metadata['Authors'],
            'Summary': doc.metadata['Summary'],  
            'Page_content': doc.page_content
        }
        temp_json2 = {
            'Title': doc.metadata['Title'],
            'Summary': doc.metadata['Summary'],  
        }
        ref_json.append(temp_json)
        result.append(temp_json2)

    with open("arxiv_docs.json", "w", encoding="utf-8") as f:
        json.dump(ref_json, f, ensure_ascii=False, indent=4)

    return result

# print(arxiv("ai in medical"))


# def call_rag(file_path=None, question="", include_arxis=False):
#     chromadb.api.client.SharedSystemClient.clear_system_cache()
#     arxiv_data = []
#     pdf_data = []
#     retriever = None
#     pdf_citation = None

#     # Process the PDF if a file path is provided
#     if file_path:
#         print("Reading PDF...")
#         loader = PyPDFLoader(file_path)
#         document = loader.load()
#         pdf_citation = file_path  # Save file name for citation

#         pdf_data = [{
#             'Title': 'Uploaded PDF',
#             'Authors': 'Unknown',
#             'Summary': document[0].page_content[:500],  # First 500 characters
#             'Page_content': document[0].page_content
#         }]

#         with open("pdf_docs.json", "w", encoding="utf-8") as f:
#             json.dump(pdf_data, f, ensure_ascii=False, indent=4)

#         # Create retriever for document content
#         chunks = chunking(document[0].page_content)
#         vector_store = Chroma.from_documents(chunks, embeddings)
#         retriever = vector_store.as_retriever(search_kwargs={"k": 1})

#     # Include arXiv data if the option is enabled
#     if include_arxis:
#         arxiv_data = arxiv(question)

#     # Set up QA chain
#     if retriever or arxiv_data:
#         print("In retrieval...")
#         combined_content = f"ArXiv content found: {arxiv_data}" if arxiv_data else ""
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever if retriever else None,
#             return_source_documents=True
#         )

#         # Generate the query
#         query = f"""
#         {combined_content}
#         Question: {question}
#         """
#         result = qa_chain.invoke({"query": query})
#         print("original result is ", result)
#         source_check = result["source_documents"][0].page_content
#         result = result['result']
#         # Match result to saved JSON for citation
#         citations = []
#         if source_check in document[0].page_content or not include_arxis:
#             citations.append({"Source": pdf_citation})
#         for data_source in arxiv_data:
#             if data_source['Summary']:
#                 citations.append({
#                     "Title": data_source.get("Title", ""),
#                     "Published": data_source.get("Published", "")
#                 })

#         # Return result with citations
#         return {
#             "result": result,
#             "citations": citations
#         }
#     else:
#         # Handle case where neither PDF nor arXiv data is available
#         return {
#             "result": "Unable to process the request. Please provide a PDF or enable arXiv data.",
#             "citations": []
#         }

def call_rag(file_path=None, question="", include_arxis=False):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    arxiv_data = []
    pdf_data = []
    retriever = None
    pdf_citation = None

    # Process the PDF if a file path is provided
    if file_path:
        print("Reading PDF...")
        loader = PyPDFLoader(file_path)
        document = loader.load()
        pdf_citation = file_path  # Save file name for citation

        pdf_data = [{
            'Title': 'Uploaded PDF',
            'Authors': 'Unknown',
            'Summary': document[0].page_content[:500],  # First 500 characters
            'Page_content': document[0].page_content
        }]

        with open("pdf_docs.json", "w", encoding="utf-8") as f:
            json.dump(pdf_data, f, ensure_ascii=False, indent=4)

        # Create retriever for document content
        chunks = chunking(document[0].page_content)
        vector_store = Chroma.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})

        # Check if the PDF is healthcare-related
        healthcare_check_prompt = f"Is the following document related to healthcare? Answer yes or no.\n\n{document[0].page_content[:500]}"
        healthcare_check_response = llm.invoke(healthcare_check_prompt)
        print(healthcare_check_response.content)
        healthcare_check_response = healthcare_check_response.content
    
        if "no" in healthcare_check_response.lower():
            return {
                "result": "This PDF is not supported as it is not related to healthcare or diagnosis.",
                "citations": []
            }

        # If related to healthcare, get character details
        character_prompt = f"""
        Based on this document's content, what would be the best character and speaking style to interact with a person diagnosed as described?
        \n\n{document[0].page_content[:500]}
        """
        character_details =  llm.invoke(character_prompt)
        pdf_data[0]['Character_Details'] = character_details

    # Include ArXiv data in the prompt if enabled
    if include_arxis:
        arxiv_data = arxiv(question)

    # Set up QA chain
    if retriever or arxiv_data:
        print("In retrieval...")
        arxiv_content = "\n".join([f"Title: {data['Title']}, Summary: {data['Summary']}" for data in arxiv_data]) if arxiv_data else ""
        combined_prompt = f"""
        ArXiv Data:
        {arxiv_content}

        PDF Character Details:
        {pdf_data[0].get('Character_Details', '') if pdf_data else ''}

        Question:
        {question}
        """

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever if retriever else None,
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": combined_prompt})
        print("original result is ", result)
        source_check = result["source_documents"][0].page_content
        result = result['result']

        # Match result to saved JSON for citation
        citations = []
        if source_check in document[0].page_content or not include_arxis:
            citations.append({"Source": pdf_citation})
        for data_source in arxiv_data:
            if data_source['Summary']:
                citations.append({
                    "Title": data_source.get("Title", ""),
                    "Published": data_source.get("Published", "")
                })

        # Return result with citations
        return {
            "result": result,
            "citations": citations
        }
    else:
        # Handle case where neither PDF nor ArXiv data is available
        return {
            "result": "Unable to process the request. Please provide a PDF or enable ArXiv data.",
            "citations": []
        }




# print("result is ",call_rag("mohit_ds.pdf","what projects are related to ai in medical and also what extra can you retrieve",include_arxis=True))




# def arxiv(topic):
#     print("retrieving research")
#     loader = ArxivLoader(
#     query=topic,
#     load_max_docs=2,
#     # doc_content_chars_max=1000,
#     # load_all_available_meta=False,
#     # ...
# )
#     docs = loader.load()
#     print(docs)
#     with open("ex.txt","w",encoding="utf-8") as f:
#         f.write(str(docs))
#     ref_json = []
#     for doc in docs:
#         temp_json = {
#             'Published': getattr(doc.metadata, 'Published', ''),
#             'Title': getattr(doc.metadata, 'Title', ''),
#             'Authors': getattr(doc.metadata, 'Authors', ''),
#             'Summary': doc.page_content[:500],  # Short summary
#             'Page_content': doc.page_content
#         }
#         ref_json.append(temp_json)
#     with open("arxiv_docs.json", "w", encoding="utf-8") as f:
#         json.dump(ref_json, f, ensure_ascii=False, indent=4)
#     return ref_json
       
# print(arxiv())

# def call_rag(file_path,question, include_arxis=False):
#     loader = PyPDFLoader(file_path)
#     document = loader.load()
#     # print(document)

#     chunks = chunking(document[0].page_content)
#     vector_store = Chroma.from_documents(chunks, embeddings)
#     # vector_store.save_local("chroma_index")
#     # docsearch = Chroma.load_local("faiss_index", embeddings)
#     retriever = vector_store.as_retriever(search_kwargs={"k": 1})
#     # print("retriever got us ", retriever)

#     if include_arxis:
#         arxiv_data = arxiv(question)

#     qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                           chain_type="stuff",
#                                           retriever=retriever,
#                                           return_source_documents=True)
#     result = qa_chain.invoke({"query": f"""arxis content found is :{arxiv_data}"" and question is {question}"""})
#     # print("result is , ",result)
#     return result["result"]



# def call_rag(file_path=None, question="", include_arxis=False):
#     arxiv_data = None
#     retriever = None

#     # Process the PDF if a file path is provided
#     if file_path:
#         print("reading pdf")
#         loader = PyPDFLoader(file_path)
#         document = loader.load()
#         chunks = chunking(document[0].page_content)
#         vector_store = Chroma.from_documents(chunks, embeddings)
#         retriever = vector_store.as_retriever(search_kwargs={"k": 1})
#     else:
#         st.warning("No PDF file provided. Only arXiv data will be used if selected.")

#     # Include arXiv data if the option is enabled
#     if include_arxis:
#         arxiv_data = arxiv(question)

#     # Set up QA chain
#     if retriever or arxiv_data:
#         print("in retrieval")
#         # Use retriever and/or arXiv data in the query
#         combined_content = f"ArXiv content found: {arxiv_data}" if arxiv_data else ""
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever=retriever if retriever else None,
#             return_source_documents=True
#         )
#         # Generate the query
#         query = f"""
#         {combined_content}
#         Question: {question}
#         """
#         result = qa_chain.invoke({"query": query})
#         print(result)
#         return result["result"]
#     else:
#         # Handle case where neither PDF nor arXiv data is available
#         return "Unable to process the request. Please provide a PDF or enable arXiv data."
