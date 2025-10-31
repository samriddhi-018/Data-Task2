
import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@st.cache_resource
def load_data_and_models():
    df = pd.read_csv('airline_faq(in).csv')
    embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    df["embedding"] = df["Question"].apply(lambda x: embedder.encode(x))
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return df, embedder, tokenizer, model

df, embedder, tokenizer, model = load_data_and_models()


def retrieve_top_k(query, df, top_k=3):
    query_vec = embedder.encode(query)
    similarities = df["embedding"].apply(lambda x: cosine_similarity([query_vec], [x])[0][0])
    top_docs = df.iloc[np.argsort(similarities)[-top_k:][::-1]]
    return top_docs

def build_prompt(query, retrieved_df):
    context = "\n".join(
        [f"Q: {row['Question']}\nA: {row['Answer']}" for _, row in retrieved_df.iterrows()]
    )
    prompt = f"""
You are Aurora Airlines' official customer support assistant.

Use *only* the provided information to answer the question below.
If the information is not present, reply:
"I’m sorry, I don’t have that information in Aurora Airlines' current policy."

Your answer must start with “Aurora Airlines” and be 2–3 lines long.

Context: {context}

Question: {query}
Answer:
"""
    return prompt


def validate_answer(answer, retrieved_df):
    context_text = " ".join(retrieved_df["Answer"].tolist()).lower()
    overlap = sum(word in context_text for word in answer.lower().split())
    if overlap < 3:
        return "I’m sorry, I don’t have that information in Aurora Airlines' current policy."
    return answer


st.set_page_config(page_title="Aurora Airlines FAQ Assistant", layout="centered")
st.title("Aurora Airlines FAQ Assistant")

query = st.text_input("Ask a question about Aurora Airlines:")

if query:
    with st.spinner("Retrieving relevant information..."):
        retrieved_df = retrieve_top_k(query, df)
        prompt = build_prompt(query, retrieved_df)

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=100)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        validated_answer = validate_answer(answer, retrieved_df)
    
    st.subheader("Answer")
    st.write(validated_answer)
    with st.expander("Retrieved FAQs"):
        for i, row in retrieved_df.iterrows():
            st.markdown(f"**Q:** {row['Question']}\n\n**A:** {row['Answer']}")


