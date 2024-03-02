# load persistent chroma instance, load pdfs into it
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from pathlib import Path
from tqdm import tqdm

import get_api_key

openai_ef = OpenAIEmbeddings()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name) #can also use model name
    num_tokens = len(encoding.encode(string))
    return num_tokens


#load PDF

def load_and_split_docs(path : Path, calc_costs = True):

    # If folder: search for all pdf's and add their paths to a list.
    if path.is_dir():
        pdfs = list(path.glob('**/*.pdf'))
        ### DEBUG
        # pdfs = pdfs[:5]
        # print('WARNING: DEBUG PDF CUTOFF ACTIVE')


    elif path.is_file():
        pdfs = [path]
        

    print(f'Loading and splitting {len(pdfs)} pdfs:')

    character_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n',"\n", ". ", " ", ""],
        chunk_size = 2048,
        chunk_overlap = 0
    )

    all_splits = []
    for pdf_path in tqdm(pdfs):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load_and_split()

        pdf_splits = character_splitter.split_documents(pages)
        all_splits += pdf_splits

    print(f'Done: Created {len(all_splits)} splits')
    if calc_costs:
        calculate_embedding_costs(all_splits)

    return all_splits

def embed_from_splits(splits):
    """Embeds all splitted documents and loads them into Chroma"""
    chroma_client = Chroma.from_documents(splits, embedding=openai_ef, persist_directory="./db")
    return chroma_client

def calculate_embedding_costs(splits): 
    """Check how many files are in the folder and how many tokens it costs to embed."""
    amount_of_tokens_to_encode = sum([num_tokens_from_string(page.page_content, 'cl100k_base') for page in tqdm(splits)])
    cost_per_1000_tokens = 0.0001 #dollar
    print(f'Tokens to encode: {amount_of_tokens_to_encode}')
    print(f'Assuming costs are ${cost_per_1000_tokens} / 1000 tokens')
    print(f'Which will cost: ${cost_per_1000_tokens*amount_of_tokens_to_encode/1000:.2}')

def get_chroma_client(from_docs : Path = None, info=True):
    """Gets a chroma client to use for RAG"""
    if from_docs:
        print('Loading Chroma from documents.')
        splits = load_and_split_docs(from_docs)
        print('Embedding splits:')
        client = embed_from_splits(splits)

    else:
        print('Loading Chroma from persistent directory')
        client = Chroma(persist_directory="./db", embedding_function=openai_ef)


    if info:
        db_info = client.get()
        num_docs = len(db_info['ids'])
        sources = set([item['source'] for item in db_info['metadatas']])
        sources = [Path(source).name for source in sources] 
        joined_sources = '\n\t'.join(sources)# Prevents error in f string
        print(f'Client contains {num_docs} documents')
        print(f'Loaded files are:\n\t{Path(joined_sources).name}')

    return client

if __name__ == '__main__':
    # client = get_chroma_client(from_docs=Path('[Your Manual Dir]'))
    client = get_chroma_client()
    print(client.get())