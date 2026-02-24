from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the SVD function from Scikit-Learn

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """    
    

    svd = TruncatedSVD(n_components=k, n_iter=10)
    M_reduced = svd.fit_transform(M) 

    return M_reduced

def plot_embeddings(M_reduced, words, user_word):
    """ Plot the embeddings. Returns a Matplotlib Figure for Streamlit. """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, word in enumerate(words):
        x_c, y_c = M_reduced[i]
        
        ax.scatter(x_c, y_c, marker='x', color='red')
        
        text_color = "red" if word == user_word else "green"
        
        ax.annotate(word, 
                    xy=(x_c, y_c),
                    xytext=(5, 5),            
                    textcoords="offset points", 
                    fontsize=12,
                    color=text_color) 

    ax.set_title(f"SVD 2D Projection for '{user_word}'")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=1.5) 
    ax.axvline(0, color='black', linewidth=1.5) 
    
    ax.set_xlim(left=-1)
    
    return fig