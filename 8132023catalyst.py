import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import numpy as np
import itertools
import openai
import re
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import tiktoken

SEC_API_KEY = "09587f5492da2c54e2175dda1b8bfaea4f3486a4f9f131656ec170a6bbb93c27"
NEWSFILTER_API_KEY = "26c64fd9c7444d23a1729d51f78e3e4cc0e2f6c594284c1f8938137ebc6bcd96"
PINECONE_API_KEY = "7348ef3c-81cb-4d25-80ad-e1ac4a5a6076"
API_ENDPOINT = "https://api.newsfilter.io/search?token={}".format(NEWSFILTER_API_KEY)
openai_api_key = "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"

os.environ["OPENAI_API_KEY"] = openai_api_key
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_articles(symbols, count=10):
    queryString = "symbols:{} AND publishedAt:[now-100d/d TO *] AND (source.id:prNewswire OR source.id:businessWire OR source.id:AccessWire OR source.id:globenewswire OR source.id:Reuters OR source.id:earningsCallTranscripts OR (source.id:sec-api AND (title:\"10-K\" OR title:\"10-Q\" OR title:\"8-K\" OR title:\"6-K\" OR title:\"20-F\")))".format(symbols)
    payload = {
        "queryString": queryString,
        "from": 0,
        "size": count
    }
    response = requests.post(API_ENDPOINT, json=payload)
    
    # Check if the response is HTML
    if response.text.startswith("<html>"):
        print("Received an HTML response:", response.text)
        return {}  # Return an empty dictionary or handle this situation as appropriate for your application
    
    articles = response.json()
    print("Newsfilter API response:", articles)
    return articles


def parse_source_url(newsfilter_article):
    source_url = newsfilter_article.get('sourceUrl')
    filing_type = newsfilter_article.get('details', {}).get('type')
    print(f"Parsing URL: {source_url} for filing type: {filing_type}")
    response = requests.get(source_url)

    # Check if the response is HTML
    if response.text.startswith("<html>"):
        print(f"Received an HTML response for URL {source_url}:", response.text)
        return "Received unexpected HTML response."  # Handle this situation as appropriate for your application

    # ... rest of your function

    if "sec.gov" in source_url:
        if filing_type in ["10-K", "10-Q"]:
            text = extract_filing_content(source_url, filing_type, SEC_API_KEY)
        elif "sec.gov" in source_url and filing_type in ["8-K", "20-F", "6-K"]:
            parts = source_url.split("/")
            cik = parts[-3]
            accession_number = parts[-2]
            filename = parts[-1]
            download_url = f"https://archive.sec-api.io/{cik}/{accession_number}/{filename}?token={SEC_API_KEY}"
            response = requests.get(download_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator="\n")
        else:
            text = "Unhandled SEC filing type"
    else:
        # Handling non-SEC URLs
        response = requests.get(source_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text(separator="\n")
    return text



def extract_filing_content(sourceUrl, filing_type, token):
    EXTRACTOR_API_ENDPOINT = "https://api.sec-api.io/extractor"
    
    items_to_extract = {
        "10-K": ["1", "1A", "7", "8"],
        "10-Q": ["part1item1", "part1item2", "part2item1", "part2item1A", "part2item2", "part2item5"],
    }
    if filing_type not in items_to_extract:
        print(f"Unsupported filing type {filing_type}")
        return ""
    extracted_content = ""
    for item in items_to_extract[filing_type]:
        params = {
            'url': sourceUrl,
            'item': item,
            'type': 'text',
            'token': SEC_API_KEY,
        }
        response = requests.get(EXTRACTOR_API_ENDPOINT, params=params)
        if response.text.startswith("<html>"):
            print(f"Received an HTML response for URL {EXTRACTOR_API_ENDPOINT}:", response.text)
            return "Received unexpected HTML response."  # Handle this situation as appropriate for your application
        if response.status_code != 200:
            print(f"Error extracting item {item} from {sourceUrl}")
            continue
        extracted_content += response.text + "\n\n"
    return extracted_content

def split_text(text):
    if not text:
        return []
    texts = text_splitter.split_text(text)
    if len(texts) == 0:
        texts = [text]  # If the text was not split, return a list containing the original text
    return texts

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def add_to_pinecone_index(vectors, index, namespace='TUP'):
    for vector_chunk in chunks(vectors, batch_size=100):
        index.upsert(vectors=vector_chunk, namespace=namespace)

def get_vectors_and_metadata(all_texts, all_filing_urls, articles, embeddings):
    metadata_list = []
    all_doc_ids = []
    all_doc_embeddings = []
    for article, filing_url in zip(articles["articles"], all_filing_urls):
        source_text = parse_source_url(article)  # Corrected this line
        split_texts = split_text(source_text)
        for i, chunk_text in enumerate(split_texts):
            doc_embeddings = embeddings.embed_documents([chunk_text])
            all_doc_embeddings.extend(doc_embeddings)
            doc_id = f"{article['id']}_part_{i}"
            all_doc_ids.append(doc_id)
            metadata_list.append(
                {
                    "text": chunk_text,
                    "filing_url": filing_url,
                    "symbols": article["symbols"],
                    "title": article["title"],
                    "sourceUrl": article["sourceUrl"],
                    "publishedAt": article["publishedAt"],
                }
            )
    return all_doc_s, all_doc_ids, metadata_list

def upsert_s_to_pinecone(index_name, doc_s, all_doc_ids, metadata_list, PINECONE_API_KEY, namespace):
    pinecone.init(api_key=PINECONE_API_KEY, environment="asia-southeast1-gcp")
    index = pinecone.Index(index_name)
    vectors = [{"id": doc_id, "values": , "metadata": metadata} for doc_id, , metadata in zip(all_doc_ids, doc_s, metadata_list)]
    add_to_pinecone_index(vectors, index, namespace=namespace)

def upload_vectors_to_pinecone(symbol):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    symbol = symbol.upper()
    articles = get_articles(symbol)
    all_texts = []
    all_filing_urls = []
    for article in articles["articles"]:
        url = article["sourceUrl"]
        text = parse_source_url(article)
        if "Error" in text:
            print(f"Error in the text: {text}")
            continue
        split_texts = split_text(text)
        all_texts.extend(split_texts)
        all_filing_urls.extend([url] * len(split_texts))
    doc_embeddings, all_doc_ids, metadata_list = get_vectors_and_metadata(all_texts, all_filing_urls, articles, embeddings)
    upsert_embeddings_to_pinecone("sec-filings", doc_embeddings, all_doc_ids, metadata_list, PINECONE_API_KEY, namespace=symbol)
    return "Embeddings for {} uploaded to Pinecone successfully!".format(symbol)

def get_openai_embedding(text):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embedding = embeddings.embed_documents([text])[0]
    return embedding

def search_pinecone_index(query, n=50, namespace=None):
    query_embedding = get_openai_embedding(query)
    index = pinecone.Index(index_name='sec-filings')
    results = index.query(vector=query_embedding, top_k=n, include_values=False, include_metadata=True, namespace=namespace)
    return results["matches"]


def filter_similar_results(results, similarity_threshold=0.9):
    filtered_results = []
    result_embeddings = []
    for result in results:
        text = result['metadata']['text']
        embedding = get_openai_embedding(text)
        if not result_embeddings:
            result_embeddings.append(embedding)
            filtered_results.append(result)
        else:
            similarities = cosine_similarity(np.array(result_embeddings), np.array(embedding).reshape(1, -1))
            max_similarity = np.max(similarities)
            if max_similarity < similarity_threshold:
                result_embeddings.append(embedding)
                filtered_results.append(result)
    return filtered_results

def summarize_text(results, model="gpt-4", max_tokens=1500, temperature=0.0):
    top_texts = " ".join([result['metadata']['text'] for result in results[:50]])
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at finding upcoming timelines, events or key dates for companies in SEC filings"},
            {"role": "user", "content": f"Based on the extracted parts of the SEC filings delimited by ---, what specific events are they expecting or anticipating will happen in 2023? Please identify forward looking statements made by the company that reference very specific initiatives, plans, or expectations and very specific dates, timelines or timeframes.  You are looking for very specific forward looking statements with very specific dates or timelines, not general forward looking statements with general timelines or timframes. Here are some examples (not all inclusive) of GENERAL forward looking statements, which do not reference specific initiatives or specific dates or timelines, which you SHOULD NOT use: 1.) 'We expect to incur significant expenses for the foreseeable future.' 2.) We anticipate delivering utmost shareholder return 3.) 'We plan to develop and commercialize new products and services' 4.) 'We expect to review our executive compensation policies'.  Here are some examples (not all inclusive) of SPECIFIC forward looking statements, which reference specific initiatives and specific dates, timeframes, or timelines, which you SHOULD use: 1.) 'We expect to begin 20 new projects in Instanbul, Turkey later this year.'2.) 'We  plan to integrate with OpenAI in the second quarter of 2023.' 3.) 'We anticipate data from the AAHG trial in the second half of 2023.' 4.) 'We plan to release a product called SWAYZEE in the coming months'. 5.) 'we believe we will announce our highest revenue quarter ever next quarter'. 6.) 'We'll receive an extension on the GHQ Loan if we establish a new partership by December 31st 2023' 7.) In the event that the Company is unable to secure additional financing by mid August 2023, it will be unable to fund its operations and may be required to evaluate further alternatives. 8.)  Nasdaq Hearings Panel has granted the Company’s request for an extension through November 20, 2023. -END OF EXAMPLES- Here are the extracted parts of the SEC filings, please create a numbered list of the specific forward looking events ---{top_texts}---"}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message["content"].strip()

def main_function(symbol):
    symbol = symbol.upper()  # Convert the ticker symbol to uppercase
    upload_vectors_to_pinecone(symbol)
    query = "what specific events are they expecting or anticipating will happen in 2023?"
    n = 50
    res = search_pinecone_index(query, n=n, namespace=symbol)
    filtered_res = filter_similar_results(res)
    results_text = []
    for match in filtered_res:
        text = match['metadata']['text'] if 'metadata' in match and 'text' in match['metadata'] else "No text found"
        text = re.sub(r'\n{3,}', '\n\n', text)
        symbols = match['metadata']['symbols'] if 'metadata' in match and 'symbols' in match['metadata'] else "No symbols found"
        title = match['metadata']['title'] if 'metadata' in match and 'title' in match['metadata'] else "No title found"
        sourceUrl = match['metadata']['sourceUrl'] if 'metadata' in match and 'sourceUrl' in match['metadata'] else "No sourceUrl found"
        industries = match['metadata']['industries'] if 'metadata' in match and 'industries' in match['metadata'] else "No industries found"
        publishedAt = match['metadata']['publishedAt'] if 'metadata' in match and 'publishedAt' in match['metadata'] else "No publishedAt found"
        results_text.append(f"Item ID: {match['id']}\nSimilarity Score: {match['score']}\nText: {text}\nSymbols: {symbols}\nTitle: {title}\nSourceUrl: {sourceUrl}\nIndustries: {industries}\nPublishedAt: {publishedAt}\n")
    openai.api_key = openai_api_key
    answer = summarize_text(filtered_res)
    return '\n'.join(results_text) + f"\n\nAnswer from OpenAI: {answer}"

st.title("Ask Edgar")

symbol = st.text_input("Enter Ticker:")
if st.button("Search"):
    result = main_function(symbol)
    st.text_area("Scroll Down to bottom for Results", value=result, height=400)


