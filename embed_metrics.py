import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np
import torch
from tqdm import tqdm

# Heat map of the cosine similarity matrix. Ideally groups should have high similarity. 
def plot_similarity_heatmap(word_grid, sim_matrix):
    words = word_grid.flatten()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        xticklabels=words,
        yticklabels=words,
        cmap="BuPu",
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title("Pairwise Cosine Similarity of Word Embeddings")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Plots the embeddings on 2D after reducing the dimensionality. Ideally the groups should be close together, but in 2D hard to say.
def plot_embeddings_2d(word_grid, embeddings, method='tsne'):
    words = word_grid.flatten()
    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(flat_embeddings)

    # Plot
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green', 'purple']
    for i in range(4):
        idxs = list(range(i*4, (i+1)*4))
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=f"Group {i+1}", color=colors[i])
        for idx in idxs:
            plt.text(reduced[idx, 0], reduced[idx, 1], words[idx], fontsize=9)

    plt.legend()
    plt.title(f"Word Embeddings 2D Projection ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Compares the distribution of in group and out group similarities. Generally out group are much more spread out.
def plot_similarity_distributions(intra_group_sims, out_group_sims):
    data = (
        [("Intra-group", sim) for sim in intra_group_sims] +
        [("Out-group", sim) for sim in out_group_sims]
    )
    labels, sims = zip(*data)
    sns.violinplot(x=labels, y=sims, inner="box", palette="pastel", hue=labels, legend=False)
    plt.title("Distribution of Cosine Similarities")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.show()

def compute_group_metrics(model, tokenizer, word_grid, plot = False):
    embeddings = np.zeros((4, 4, model.config.hidden_size))

    # Generate Embeddings with model
    for i in range(4):
        for j in range(4):
            word = word_grid[i, j]
            embeddings[i, j] = get_word_embedding(model, tokenizer, word)

    flat_embeddings = embeddings.reshape(-1, embeddings.shape[-1])  # (16, hidden_dim)
    sim_matrix = cosine_similarity(flat_embeddings)

    # Intra-group similarity
    intra_group_sims = []
    for group_idx in range(4):
        idxs = [group_idx * 4 + i for i in range(4)]
        group_sims = sim_matrix[np.ix_(idxs, idxs)]
        upper_triangle = group_sims[np.triu_indices(4, k=1)]
        intra_group_sims.append(upper_triangle.mean())

    # Out-group similarity
    out_group_sims = []
    for i in range(16):
        group_i = i // 4
        for j in range(16):
            group_j = j // 4
            if group_i != group_j:
                out_group_sims.append(sim_matrix[i, j])

    if plot:
        plot_embeddings_2d(word_grid, embeddings)
        plot_similarity_heatmap(word_grid, sim_matrix)
        plot_similarity_distributions(intra_group_sims, out_group_sims)

    used = set()
    groups = []

    # Greedy grouping of the words based on similarity
    all_indices = list(range(16))
    while len(used) < 16:
        best_group = None
        best_score = -float('inf')
        for combo in itertools.combinations([i for i in all_indices if i not in used], 4):
            score = sum(sim_matrix[i][j] for i, j in itertools.combinations(combo, 2))
            if score > best_score:
                best_score = score
                best_group = combo
        groups.append(best_group)
        used.update(best_group)

    words_flat = word_grid.flatten()

    return {
        "avg_intra_group_similarity": np.mean(intra_group_sims),
        "intra_group_similarities": intra_group_sims,
        "avg_out_group_similarity": np.mean(out_group_sims),
        "group_assignments": [[words_flat[idx] for idx in group] for group in groups],
    }

def get_word_embedding(model, tokenizer, word):
    # Tokenize the word
    inputs = tokenizer(word, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]

    # Get the hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Choose the last hidden layer
        last_hidden_state = outputs.hidden_states[-1]

    # Each token has its own vector; average across tokens if the word was split
    word_embedding = last_hidden_state[0].mean(dim=0)  # shape: (hidden_dim,)
    return word_embedding.cpu().numpy()

def getEmbedMetric(words, model, tokenizer):
    word_grid = np.array(words).reshape((4, 4))
    metrics = compute_group_metrics(model, tokenizer, word_grid, plot=True)

    print("Average Intra-Group Similarity:", metrics["avg_intra_group_similarity"])
    for idx, sim in enumerate(metrics["intra_group_similarities"]):
        print(f"Group {idx+1} similarity: {sim:.4f}")

    print("Average Out-Group Similarity:", metrics["avg_out_group_similarity"])

    print("Greedy Groupings:")
    for i, group in enumerate(metrics["group_assignments"]):
        print(f"Group {i+1}: {group}")

def getAvgEmbedMetric(wordLists, model, tokenizer, ckpt_end_path):
    word_grids = [np.array(words).reshape((4, 4)) for words in wordLists]
    metrics = []
    for word_grid in tqdm(word_grids, desc=ckpt_end_path):
        metrics.append(compute_group_metrics(model, tokenizer, word_grid, plot=True))
    avgIntraGrpSim = sum([metric["avg_intra_group_similarity"] for metric in metrics]) / len(wordLists)
    avgExtraGrpSim = sum([metric["avg_out_group_similarity"] for metric in metrics]) / len(wordLists)
    return avgIntraGrpSim, avgExtraGrpSim
    