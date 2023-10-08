from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

OPENAI_API_KEY = 'YOUR_API_KEY'

llm = OpenAI(openai_api_key=OPENAI_API_KEY)

doc_loader = UnstructuredFileLoader("sample.txt")
doc = doc_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

chunks = text_splitter.split_documents(doc)

chain = load_qa_chain(llm, chain_type="map_reduce", verbose=True, return_intermediate_steps=True)

def ask_question(query):
    answer = chain({"input_documents": chunks, "question": query}, return_only_outputs=True)
    return answer['output_text']

if __name__ == "__main__":
    query = "What is the Junk Fee Prevention Act?"
    answer = ask_question(query)
    print(answer)
