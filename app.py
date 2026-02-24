import numpy as np
import streamlit as st
import gensim.downloader as api
import pandas as pd
from utils import reduce_to_k_dim, plot_embeddings

st.set_page_config(page_title="GloVe Word Explorer", page_icon="🧠", layout="centered")
st.title("🧠 GloVe Semantic Explorer")
st.markdown("Exploring 400,000 word embeddings using Gensim and Streamlit.")

@st.cache_resource
def load_embedding_model():
    with st.spinner("Loading GloVe vectors (this takes a moment on first run)..."):
        wv_from_bin = api.load("glove-wiki-gigaword-200")
        return wv_from_bin

# Load the model
wv = load_embedding_model()
st.success(f"Successfully loaded {len(wv)} words!")

st.subheader("🔍 Find Semantically Similar Words")
st.write("Type a word to find its nearest neighbors in the 200-dimensional vector space.")

# Text input
user_word = st.text_input("Enter a single word (e.g. 'technology'):").lower().strip()

if user_word:
    try:
        # Gensim's built-in highly optimized cosine similarity calculator
        similar_words = wv.most_similar(user_word, topn=5)
        
        st.write(f"Top 5 closest words to **'{user_word}'**:")
        
        # Format the output nicely into a Pandas DataFrame for Streamlit to render as a table
        results_df = pd.DataFrame(similar_words, columns=["Word", "Cosine Similarity"])
        
        # Display the table
        st.table(results_df)

        st.write("Plot of word vector")
        
        words = [user_word] + [x[0] for x in similar_words]
        M = np.array([wv[w] for w in words])
        M_reduced = reduce_to_k_dim(M)

        fig = plot_embeddings(M_reduced, words, user_word)
        st.pyplot(fig)

    except KeyError:
        # If the user types a word not in the 400,000 vocabulary (like a typo)
        st.error(f"Oops! The word '{user_word}' is not in the GloVe vocabulary. Try another one.")