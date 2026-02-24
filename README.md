# 🧠 Semantic Vector Space Explorer

A full-stack, interactive data visualization dashboard that explores the geometric relationships of human language using 400,000 pre-trained GloVe word embeddings. 

Built as an applied engineering project bridging the theoretical mathematics of **Stanford's CS224n (Natural Language Processing with Deep Learning)** with modern, production-ready Python web frameworks.

## ✨ Features

* **Semantic Search Engine**: Utilizes highly optimized **Cosine Similarity** algorithms to search a 200-dimensional vector space and retrieve the top 5 nearest semantic neighbors to any given user input.
* **Dimensionality Reduction**: Implements **Truncated SVD (Singular Value Decomposition)** via Scikit-Learn to compress 200-dimensional vectors down to a 2D coordinate plane, preserving maximum variance.
* **Geometric Visualization**: Renders interactive 2D scatter plots using Matplotlib, complete with strict origin `(0,0)` axes. This allows users to visually inspect the exact angles and vector magnitudes that determine semantic distance.
* **Caching & Performance**: Employs Streamlit's `@st.cache_resource` to securely load and store the massive Gensim NLP model in local RAM, preventing memory leaks and ensuring lightning-fast UI rendering.

## 🛠️ Tech Stack

* **Frontend/UI**: Streamlit
* **Data Processing**: NumPy, Pandas
* **Machine Learning / NLP**: Gensim (GloVe-Wiki-Gigaword-200)
* **Dimensionality Reduction**: Scikit-Learn (TruncatedSVD)
* **Data Visualization**: Matplotlib

## 📊 The Math: "The Curse of Dimensionality"

A key feature of this dashboard is demonstrating the mathematical difference between true multi-dimensional distance and 2D projections. 

For example, searching the word **"house"** will reveal that **"senate"** has a very high Cosine Similarity ($0.6695$) in 200D space. However, on the 2D SVD scatter plot, they appear visually distant. This visually proves the mathematical concept of *Information Loss*—when SVD squashes 200 dimensions down to 2, we lose 198 dimensions of variance, creating a 2D "shadow" of a 200D reality. 
