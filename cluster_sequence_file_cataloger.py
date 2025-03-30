# cluster_filenames_tfidf.py

import os
import argparse
import json
import datetime
import time
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from difflib import SequenceMatcher
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import tempfile

try:
    # Try to import sentence-transformers for embedding-based naming
    from sentence_transformers import SentenceTransformer
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    print("Note: sentence-transformers not found. Install with 'pip install sentence-transformers' for enhanced sequence naming.")

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False
    print("Note: PIL/Pillow not found. Install with 'pip install Pillow' for sequence preview GIFs.")

# Define common image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.gif', '.bmp', 
    '.webp', '.heic', '.heif', '.raw', '.svg', '.psd', 
    '.cr2', '.crw', '.nef', '.arw', '.dng', '.orf', 'c4d',
    '.mp4', '.mov', '.mpg', '.mpeg', '.avi'  # Add video formats
}

def find_image_files(directory):
    """
    Recursively find all image files in the given directory.
    
    Args:
        directory: Directory path to search
        
    Returns:
        List of image file paths
    """
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file.lower())[1]
            if ext in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, file))
                
    return image_files

def get_file_metadata(filepath):
    """Get metadata for a file including size and timestamps."""
    stats = os.stat(filepath)
    return {
        "filename": os.path.basename(filepath),
        "filepath": filepath,
        "filesize": stats.st_size,
        "created": datetime.datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()
    }

def group_files_by_directory(files):
    """Group files by their parent directory."""
    directory_groups = defaultdict(list)
    
    for file in files:
        parent_dir = os.path.dirname(file)
        directory_groups[parent_dir].append(file)
    
    return directory_groups

def cluster_filenames_tfidf(filenames, eps=0.7, min_samples=3):
    """
    Cluster filenames using character-level TF-IDF and DBSCAN.
    """
    # Use only the basename of the files for clustering
    basenames = [os.path.basename(f) for f in filenames]
    
    # Skip clustering if there are too few files
    if len(basenames) < min_samples:
        return {}, filenames
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
    vectors = vectorizer.fit_transform(basenames)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(vectors)

    clusters = {}
    unclustered = []
    
    for label, filename in zip(labels, filenames):
        if label == -1:
            unclustered.append(filename)
            continue
        clusters.setdefault(label, []).append(filename)
    
    return clusters, unclustered

def similarity_ratio(str1, str2):
    """Calculate string similarity ratio using SequenceMatcher."""
    return SequenceMatcher(None, str1, str2).ratio()

def identify_sequences_by_similarity(clusters):
    """
    Identify potential sequences within clusters based on similarity patterns.
    """
    sequences = {}
    regular_clusters = {}
    
    for cluster_id, files in clusters.items():
        if len(files) < 3:  # Need at least 3 files to form a meaningful sequence
            regular_clusters[cluster_id] = files
            continue
            
        # Sort files by basename - this often naturally orders sequences
        files_sorted = sorted(files, key=os.path.basename)
        basenames = [os.path.basename(f) for f in files_sorted]
        
        # Calculate similarity between consecutive files
        consecutive_similarities = []
        for i in range(1, len(basenames)):
            sim = similarity_ratio(basenames[i-1], basenames[i])
            consecutive_similarities.append(sim)
        
        # Check if similarities are consistent (a sign of sequences)
        if len(consecutive_similarities) >= 2:
            similarities_array = np.array(consecutive_similarities)
            
            # Measure consistency: standard deviation should be low for sequences
            similarity_std = np.std(similarities_array)
            similarity_mean = np.mean(similarities_array)
            
            # Sequences typically have high mean similarity and low standard deviation
            is_likely_sequence = (similarity_mean > 0.7 and similarity_std < 0.1)
            
            if is_likely_sequence:
                sequences[f"seq_{cluster_id}"] = files_sorted
                continue
        
        # If not identified as a sequence, keep as regular cluster
        regular_clusters[cluster_id] = files
    
    return sequences, regular_clusters

def derive_sequence_name(filenames):
    """
    Derive a meaningful semantic name from sequence filenames.
    Returns a descriptive name based on common patterns in the filenames.
    """
    if not filenames:
        return "unnamed_sequence"
    
    # Extract basenames without extensions and directory paths
    basenames = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    
    # Remove numeric parts (typically sequence numbers)
    non_numeric_parts = []
    for name in basenames:
        # Replace digits with spaces
        cleaned = re.sub(r'[0-9]+', ' ', name)
        # Remove common separators and replace with spaces
        cleaned = re.sub(r'[_\-\.]', ' ', cleaned)
        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned:
            non_numeric_parts.append(cleaned)
    
    if not non_numeric_parts:
        return "unnamed_sequence"
    
    # Find common words across filenames
    word_lists = [name.split() for name in non_numeric_parts]
    
    # Flatten the list of words and count frequencies
    all_words = [word for word_list in word_lists for word in word_list if len(word) > 2]
    word_counts = Counter(all_words)
    
    # Get most common words that appear in most filenames
    common_words = [word for word, count in word_counts.most_common(3) 
                   if count >= len(filenames) * 0.8]  # Word appears in at least 80% of files
    
    if common_words:
        # Join common words to form a descriptive name
        sequence_name = "_".join(common_words)
    else:
        # If no common words, try finding the longest common substring
        longest_common = non_numeric_parts[0]
        for name in non_numeric_parts[1:]:
            matcher = SequenceMatcher(None, longest_common, name)
            match = matcher.find_longest_match(0, len(longest_common), 0, len(name))
            if match.size > 0:
                longest_common = longest_common[match.a:match.a + match.size].strip()
            else:
                longest_common = ""
                break
        
        if longest_common and len(longest_common) > 3:
            sequence_name = longest_common.replace(" ", "_")
        else:
            # Fallback: use the most frequent word if available
            if word_counts:
                sequence_name = word_counts.most_common(1)[0][0]
            else:
                # Last resort: use part of the first filename
                first_name = os.path.splitext(os.path.basename(filenames[0]))[0]
                alpha_only = re.sub(r'[^a-zA-Z]', '', first_name)
                sequence_name = alpha_only[:15] if alpha_only else "unnamed_sequence"
    
    # Clean up final name: remove any remaining special characters and normalize
    sequence_name = re.sub(r'[^a-zA-Z0-9_]', '', sequence_name)
    sequence_name = re.sub(r'_+', '_', sequence_name)
    
    return sequence_name.lower() if sequence_name else "unnamed_sequence"

def derive_sequence_name_ai(filenames, directory=None):
    """
    Generate a meaningful sequence name using AI techniques.
    
    Args:
        filenames: List of files in the sequence
        directory: Directory containing the files (optional)
        
    Returns:
        A human-readable name for the sequence
    """
    if not filenames:
        return "unnamed_sequence"
        
    # 1. Extract path components for context
    first_file = filenames[0]
    path_parts = Path(os.path.dirname(first_file)).parts
    significant_path_parts = [p for p in path_parts[-3:] if not p.startswith('.')]
    
    # 2. Extract filename components (without numbers)
    basenames = [os.path.splitext(os.path.basename(f))[0] for f in filenames]
    cleaned_names = []
    for name in basenames:
        # Replace digits with spaces
        cleaned = re.sub(r'[0-9]+', ' ', name)
        # Remove common separators and replace with spaces
        cleaned = re.sub(r'[_\-\.]', ' ', cleaned)
        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            cleaned_names.append(cleaned)
    
    # 3. Extract common words and tokens
    all_tokens = []
    
    # Add significant path tokens
    for part in significant_path_parts:
        tokens = re.split(r'[_\-\s]', part)
        all_tokens.extend([t for t in tokens if len(t) > 2])
        
    # Add filename tokens
    for name in cleaned_names:
        words = name.split()
        all_tokens.extend([w for w in words if len(w) > 2])
    
    # 4. Clean and normalize tokens
    cleaned_tokens = []
    for token in all_tokens:
        # Keep only alphabetic characters
        clean_token = re.sub(r'[^a-zA-Z]', '', token).lower()
        if clean_token and len(clean_token) > 2:
            cleaned_tokens.append(clean_token)
    
    # 5. Use AI approach if available
    if HAVE_TRANSFORMERS:
        try:
            return derive_name_with_embeddings(cleaned_tokens, cleaned_names)
        except Exception as e:
            print(f"Error using transformer model: {e}")
            # Fall back to simpler method
    
    # 6. Fall back to frequency-based approach
    token_counts = Counter(cleaned_tokens)
    
    # Extract most common meaningful tokens
    common_tokens = [token for token, count in token_counts.most_common(3) 
                   if count >= max(2, len(filenames) * 0.2)]
    
    if common_tokens:
        return "_".join(common_tokens)
        
    # Last resort - use existing method
    return derive_sequence_name(filenames)

def derive_name_with_embeddings(tokens, cleaned_names, model_name="all-MiniLM-L6-v2"):
    """Use embeddings to find the most representative terms."""
    # Skip if no tokens or names
    if not tokens or not cleaned_names:
        return "unnamed_sequence"
    
    # Deduplicate tokens first
    unique_tokens = list(set(tokens))
    
    # Load the model (this is cached so subsequent calls are fast)
    model = SentenceTransformer(model_name)
    
    # Combine tokens and full names for comparison
    all_terms = unique_tokens + cleaned_names
    
    if len(all_terms) < 2:
        return all_terms[0] if all_terms else "unnamed_sequence"
        
    # Generate embeddings
    embeddings = model.encode(all_terms)
    
    # Find centroid (center of mass of all embeddings)
    centroid = np.mean(embeddings, axis=0)
    
    # Calculate similarity to centroid
    similarities = np.dot(embeddings, centroid) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(centroid)
    )
    
    # Select top 2-3 most central tokens that are unique
    token_indices = list(range(len(unique_tokens)))  # Use unique_tokens length
    name_indices = list(range(len(unique_tokens), len(all_terms)))
    
    # Sort tokens by similarity to centroid
    sorted_token_indices = sorted(token_indices, key=lambda i: similarities[i], reverse=True)
    
    # Get most representative tokens (up to 3)
    top_tokens = []
    for idx in sorted_token_indices:
        token = all_terms[idx]
        if token not in top_tokens and len(top_tokens) < 3:
            top_tokens.append(token)
    
    if not top_tokens:
        # Try names if tokens didn't work
        sorted_name_indices = sorted(name_indices, key=lambda i: similarities[i], reverse=True)
        if sorted_name_indices:
            most_central_name = all_terms[sorted_name_indices[0]]
            words = most_central_name.split()
            top_tokens = [w for w in words if len(w) > 2][:3]
    
    if top_tokens:
        return "_".join(top_tokens)
    
    return "unnamed_sequence"

def extract_frames_from_video(video_path, max_frames=60):
    """
    Extract frames from a video file for GIF creation.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of PIL Image objects
    """
    if not HAVE_PIL:
        print(f"Cannot extract frames from {video_path}: PIL/Pillow library not available")
        return []
        
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step size to evenly sample frames
        step = max(1, frame_count // max_frames)
        
        # Extract frames
        frames = []
        for i in range(0, frame_count, step):
            if len(frames) >= max_frames:
                break
                
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB (PIL uses RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(frame_rgb)
                
                # Resize if needed
                pil_img.thumbnail((300, 300))
                
                frames.append(pil_img)
        
        # Release the video capture object
        cap.release()
        
        return frames
    
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def create_preview_gif(file_path, output_name, sequence_id):
    """
    Create a preview GIF for a file (either a video or part of an image sequence).
    
    Args:
        file_path: Path to the file (video or image)
        output_name: Base name for the output GIF
        sequence_id: Unique ID of the sequence
        
    Returns:
        Path to the created GIF file or None if creation failed
    """
    if not HAVE_PIL:
        print(f"Cannot create GIF: PIL/Pillow library not available")
        return None
    
    # Create output directory
    unique_dir_name = f"{output_name}_{sequence_id}"
    output_dir = os.path.join("seq_gifs", unique_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine file type
    ext = os.path.splitext(file_path.lower())[1]
    is_video = ext in {'.mp4', '.mov', '.mpg', '.mpeg', '.avi'}
    
    frames = []
    try:
        if is_video:
            # For video files, extract frames
            frames = extract_frames_from_video(file_path)
        else:
            # For single images, just load the image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            
            # Convert to RGB if necessary
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            frames = [img]
    
        if not frames:
            print(f"No frames generated for {file_path}")
            return None
    
        # Save as GIF
        output_path = os.path.join(output_dir, f"{output_name}_{sequence_id}.gif")
        
        # Save with a reasonable frame duration (100ms)
        frames[0].save(
            output_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=100,  # milliseconds per frame
            loop=0  # 0 means loop forever
        )
        print(f"Saved preview GIF to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating GIF from {file_path}: {e}")
        return None

def create_sequence_preview_gif(sequence_files, sequence_name, sequence_id, max_frames=60, min_frames=20, max_dimension=300):
    """
    Create a preview GIF for a sequence with a large number of files.
    
    Args:
        sequence_files: List of image files in the sequence
        sequence_name: Semantic name of the sequence
        sequence_id: Unique ID of the sequence
        max_frames: Maximum number of frames to include in the GIF
        min_frames: Minimum number of frames to include in the GIF
        max_dimension: Maximum width or height in pixels
        
    Returns:
        Path to the created GIF file or None if creation failed
    """
    if not HAVE_PIL:
        print(f"Cannot create GIF for {sequence_name}: PIL/Pillow library not available")
        return None
    
    import numpy as np
    
    # Create a unique directory name that includes both semantic name and sequence ID
    unique_dir_name = f"{sequence_name}_{sequence_id}"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("seq_gifs", unique_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of frames to use
    num_files = len(sequence_files)
    num_frames = min(max_frames, max(min_frames, num_files))
    
    # Sample files evenly across the sequence
    if num_frames >= num_files:
        selected_files = sequence_files
    else:
        indices = np.linspace(0, num_files - 1, num_frames, dtype=int)
        selected_files = [sequence_files[i] for i in indices]
    
    print(f"Creating GIF for {sequence_name} using {len(selected_files)} frames")
    
    # Process images and create GIF
    frames = []
    for file_path in selected_files:
        try:
            ext = os.path.splitext(file_path.lower())[1]
            is_video = ext in {'.mp4', '.mov', '.mpg', '.mpeg', '.avi'}
            
            if is_video:
                video_frames = extract_frames_from_video(file_path, max_frames=max_frames)
                frames.extend(video_frames)
            else:
                img = Image.open(file_path)
                
                # Resize while maintaining aspect ratio
                img.thumbnail((max_dimension, max_dimension))
                
                # Convert to RGB if necessary (for formats like PNG with alpha)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                    
                frames.append(img)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if not frames:
        print(f"No valid frames found for {sequence_name}")
        return None
    
    # Save as GIF with the unique name
    output_path = os.path.join(output_dir, f"{sequence_name}_{sequence_id}.gif")
    
    try:
        # Save with a reasonable frame duration (100ms)
        frames[0].save(
            output_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=100,  # milliseconds per frame
            loop=0  # 0 means loop forever
        )
        print(f"Saved preview GIF to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving GIF for {sequence_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Cluster image files in a directory based on filename similarity')
    parser.add_argument('--directory', '-d', type=str, default='.', 
                        help='Directory to search for image files (default: current directory)')
    parser.add_argument('--output', '-o', type=str,
                        help='Output JSON file path (default: auto-generated)')
    args = parser.parse_args()
    
    # Start tracking overall execution time
    start_total_time = time.time()
    
    directory = os.path.abspath(args.directory)
    print(f"Searching for files in: {directory}")
    
    # Track file discovery time
    start_discovery_time = time.time()
    image_files = find_image_files(directory)
    discovery_time = time.time() - start_discovery_time
    print(f"Found {len(image_files)} files in {discovery_time:.2f} seconds")
    
    if not image_files:
        print("No files found.")
        return
    
    # Create GIFs for video files
    video_gifs = {}
    video_files = []

    for filepath in image_files:
        ext = os.path.splitext(filepath.lower())[1]
        if ext in {'.mp4', '.mov', '.mpg', '.mpeg', '.avi'}:
            video_files.append(filepath)

    print(f"Found {len(video_files)} video files for GIF creation")

    for video_path in video_files:
        # Create a safe filename version of the video name
        basename = os.path.basename(video_path)
        name_part = os.path.splitext(basename)[0]
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name_part)
        
        # Use the full path hash as the sequence ID to ensure uniqueness
        sequence_id = f"video_{abs(hash(video_path)) % 10000}"
        
        # Create GIF
        gif_path = create_preview_gif(video_path, safe_name, sequence_id)
        
        if gif_path:
            video_gifs[video_path] = gif_path
    
    # Track directory grouping time
    start_grouping_time = time.time()
    directory_groups = group_files_by_directory(image_files)
    grouping_time = time.time() - start_grouping_time
    print(f"Found {len(directory_groups)} directories with files in {grouping_time:.2f} seconds")
    
    # Track clustering and sequence identification time
    start_clustering_time = time.time()
    all_sequences = {}
    all_clusters = {}
    all_unclustered = []
    
    for dir_path, files in directory_groups.items():
        print(f"Processing directory: {dir_path} ({len(files)} files)")
        
        # Cluster files in this directory
        dir_clusters, dir_unclustered = cluster_filenames_tfidf(files)
        
        # Identify sequences in this directory's clusters
        dir_sequences, dir_regular_clusters = identify_sequences_by_similarity(dir_clusters)
        
        # Add unique identifiers to incorporate directory information
        dir_id = os.path.basename(dir_path)
        
        # Add to global collections with unique IDs
        for seq_id, seq_files in dir_sequences.items():
            all_sequences[f"{dir_id}_{seq_id}"] = seq_files
            
        for cluster_id, cluster_files in dir_regular_clusters.items():
            all_clusters[f"{dir_id}_{cluster_id}"] = cluster_files
            
        all_unclustered.extend(dir_unclustered)
    
    clustering_time = time.time() - start_clustering_time
    print(f"Clustering and sequence identification completed in {clustering_time:.2f} seconds")
    
    # Display summary results to console
    print("\n=== SUMMARY ===")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"Total non-sequence clusters: {len(all_clusters)}")
    
    total_in_sequences = sum(len(s) for s in all_sequences.values())
    total_in_clusters = sum(len(c) for c in all_clusters.values())
    print(f"Files in sequences: {total_in_sequences}")
    print(f"Files in other clusters: {total_in_clusters}")
    print(f"Unclustered files: {len(all_unclustered)}")
    
    # Semantic name generation for sequences
    start_naming_time = time.time()
    sequence_names = {}
    for seq_id, files in all_sequences.items():
        sequence_names[seq_id] = derive_sequence_name_ai(files)
        print(f"Derived sequence name: {sequence_names[seq_id]} for {len(files)} files")
    naming_time = time.time() - start_naming_time
    print(f"Sequence naming completed in {naming_time:.2f} seconds")
    
    # GIF creation as a separate step
    start_gif_time = time.time()
    preview_gifs = {}
    
    for seq_id, files in all_sequences.items():
        if len(files) > 30 and HAVE_PIL:
            # Create a safe filename version of the sequence name
            semantic_name = sequence_names[seq_id]
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', semantic_name)
            
            # Pass both the semantic name and the sequence ID
            gif_path = create_sequence_preview_gif(
                files, 
                safe_name,
                seq_id.replace("/", "_").replace("\\", "_")
            )
            
            if gif_path:
                preview_gifs[seq_id] = gif_path
    
    gif_time = time.time() - start_gif_time
    print(f"GIF creation completed in {gif_time:.2f} seconds")
    
    # Now prepare JSON (without GIF creation)
    start_json_time = time.time()
    
    # Create result dictionary
    json_data = {
        "metadata": {
            "root_path": directory,
            "scan_date": datetime.datetime.now().isoformat(),
            "total_images": len(image_files),
            "total_clusters": len(all_clusters),
            "total_sequences": len(all_sequences),
            "unclustered_count": len(all_unclustered),
            "timing": {
                "total_seconds": 0,  # Will be updated at the end
                "file_discovery_seconds": discovery_time,
                "directory_grouping_seconds": grouping_time,
                "clustering_seconds": clustering_time,
                "sequence_naming_seconds": naming_time,
                "gif_creation_seconds": gif_time,
                "json_preparation_seconds": 0  # Will be updated after JSON preparation
            }
        },
        "sequences": [],
        "clusters": [],
        "unclustered": []
    }
    
    # Add sequences with AI-generated semantic names
    for seq_id, files in all_sequences.items():
        seq_data = {
            "id": seq_id,
            "name": sequence_names[seq_id],
            "directory": os.path.dirname(files[0]) if files else "",
            "count": len(files),
            "files": [get_file_metadata(f) for f in files]
        }
        
        # Add preview GIF path if available
        if seq_id in preview_gifs:
            seq_data["preview_gif"] = preview_gifs[seq_id]
            
        json_data["sequences"].append(seq_data)
    
    # Add clusters
    for cluster_id, files in all_clusters.items():
        cluster_data = {
            "id": str(cluster_id),
            "directory": os.path.dirname(files[0]) if files else "",
            "count": len(files),
            "files": [get_file_metadata(f) for f in files]
        }
        json_data["clusters"].append(cluster_data)
    
    # Add unclustered files
    json_data["unclustered"] = [get_file_metadata(f) for f in all_unclustered]
    
    json_time = time.time() - start_json_time
    
    # Update JSON preparation time in timing data
    json_data["metadata"]["timing"]["json_preparation_seconds"] = json_time
    
    # Update total time
    total_time = time.time() - start_total_time
    json_data["metadata"]["timing"]["total_seconds"] = total_time
    
    print(f"JSON preparation completed in {json_time:.2f} seconds")
    
    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    output_file = args.output if args.output else f"files_clustering_{timestamp}.json"
    
    # Save JSON output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
