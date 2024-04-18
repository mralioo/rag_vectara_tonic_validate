from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os


documents = SimpleDirectoryReader("./paul_graham_essays").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Gets the response from llama index
def get_llama_response(prompt):
    response = query_engine.query(prompt)
    context = [x.text for x in response.source_nodes]
    return {
        "llm_answer": response.response,
        "llm_context_list": context
    }