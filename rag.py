from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from vector_db import get_chroma_client

llm = ChatOpenAI()

chroma_client = get_chroma_client(info=False)

query = "Reccommend a nice piano for my next project" 

retriever = chroma_client.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | llm | output_parser

# print(chain.invoke(query))

if __name__ == '__main__':
    while True:
        query = input('Give your question to MusicManualQA:\n> ')
        print(chain.invoke(query))