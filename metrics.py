import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

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

def avg_plot_similarity_heatmap(sim_matrix):
    words = word_grid.flatten()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        cmap="BuPu",
        square=True,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Cosine Similarity"}
    )
    plt.title("Average Pairwise Cosine Similarity of Word Embeddings")
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
    sns.violinplot(x=labels, y=sims, inner="box", palette="pastel")
    plt.title("Distribution of Cosine Similarities")
    plt.ylabel("Cosine Similarity")
    plt.tight_layout()
    plt.show()


#call this with either string numpy array wordgrids of shape (n, 4, 4)
# if n=1 it will output the groupings and all plots
# if n>1 it will output the plots that make sense
def compute_group_metrics(model, tokenizer, word_grids, plot=False):
    # Ensure input is (n, 4, 4)
    if word_grids.ndim == 2:
        word_grids = word_grids[np.newaxis, :, :]
    
    num_games = word_grids.shape[0]
    hidden_size = model.config.hidden_size

    all_intra_sims = []
    all_out_sims = []
    all_sim_matrices = []

    for game_idx in range(num_games):
        word_grid = word_grids[game_idx]
        embeddings = np.zeros((4, 4, hidden_size))

        for i in range(4):
            for j in range(4):
                word = word_grid[i, j]
                embeddings[i, j] = get_word_embedding(model, tokenizer, word)

        flat_embeddings = embeddings.reshape(-1, hidden_size)  # (16, hidden_dim)
        sim_matrix = cosine_similarity(flat_embeddings)
        all_sim_matrices.append(sim_matrix)

        # Intra-group similarity (assume group 0 is indices 0–3, group 1 is 4–7, etc.)
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

        all_intra_sims.append(intra_group_sims)
        all_out_sims.append(out_group_sims)

    # Convert to numpy arrays
    all_intra_sims = np.array(all_intra_sims)  # (n_games, 4)
    all_out_sims = np.array(all_out_sims)      # (n_games, num_out_pairs)

    avg_intra = all_intra_sims.mean(axis=0)
    avg_out = all_out_sims.mean()

    # Plotting
    if plot:
        avg_sim_matrix = np.mean(np.array(all_sim_matrices), axis=0)
        if num_games == 1:
            plot_embeddings_2d(word_grids[0], embeddings)
            plot_similarity_heatmap(word_grids[0], avg_sim_matrix)
            plot_similarity_distributions(avg_intra, all_out_sims.flatten())
        else:
            avg_plot_similarity_heatmap(avg_sim_matrix)
            plot_similarity_distributions(avg_intra, all_out_sims.flatten())

    result = {
        "avg_intra_group_similarity": avg_intra.mean(),
        "intra_group_similarities": avg_intra.tolist(),
        "avg_out_group_similarity": float(avg_out),
    }

    # Group assignments only if single game
    if num_games == 1:
        word_grid = word_grids[0]
        sim_matrix = all_sim_matrices[0]
        used = set()
        groups = []
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
        result["group_assignments"] = [[words_flat[idx] for idx in group] for group in groups]

    return result

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


#compute_group_metrics(model, tokenizer, word_grids, plot=False) for metrics

# words = [[
#             "CITRUS",
#             "LEAFY GREENS",
#             "SUNSHINE",
#             "SUPPLEMENTS",
#             "CITY",
#             "LAND",
#             "TOWN",
#             "WORLD",
#             "AMERICAN FLAG",
#             "GALAXY",
#             "RED CARPET",
#             "UBER RATING",
#             "ALL OUT",
#             "BETWEEN",
#             "KART",
#             "STEADY"
#         ]]

# word_grid = np.array(words).reshape((len(words), 4, 4))
# metrics = compute_group_metrics(model, tokenizer, word_grid, plot=True)

# print("Average Intra-Group Similarity:", metrics["avg_intra_group_similarity"])
# for idx, sim in enumerate(metrics["intra_group_similarities"]):
#     print(f"Group {idx+1} similarity: {sim:.4f}")

# print("Average Out-Group Similarity:", metrics["avg_out_group_similarity"])

# if len(words) == 1:
#     print("Greedy Groupings:")
#     for i, group in enumerate(metrics["group_assignments"]):
#         print(f"Group {i+1}: {group}")