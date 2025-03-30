from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import os
import numpy as np  # Import NumPy for tensor conversion

def detect_sequences(filenames, model_name="all-MiniLM-L6-v2"):
    """
    Detect sequences of filenames using embeddings and clustering.
    """
    # Load a pre-trained sentence-transformer model
    model = SentenceTransformer(model_name)

    # Compute embeddings for filenames
    embeddings = model.encode(filenames, convert_to_tensor=True)

    # Convert embeddings to a NumPy array on the CPU
    embeddings = embeddings.cpu().numpy()

    # Apply DBSCAN to find clusters
    clustering = DBSCAN(eps=0.2, min_samples=3, metric="cosine").fit(embeddings)

    # Group filenames by cluster labels
    clusters = {}
    for label, filename in zip(clustering.labels_, filenames):
        if label == -1:
            continue  # Ignore noise
        clusters.setdefault(label, []).append(filename)

    return clusters

# Example usage
if __name__ == "__main__":
    # Simulated filenames in a folder
    filenames = [
        "image_a.jpg", "image_b.jpg", "image_c.jpg",
        "frame_001.png", "frame_002.png", "frame_003.png",
        "random.jpg", "README.md", "foo.md", "example.md", "photo_o.jpeg", "photo_x.jpeg", "photo_y.jpeg"
    ]

    clusters = detect_sequences(filenames)
    print("Detected Clusters:")
    for cluster_id, files in clusters.items():
        print(f"Cluster {cluster_id}: {files}")
