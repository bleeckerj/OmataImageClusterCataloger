# Omata Image Catalog Clustering

This project provides tools for clustering and cataloging image and video files based on filename similarity and directory structure. It generates a comprehensive JSON catalog of sequences, clusters, and unclustered files, with semantic sequence naming and preview GIFs for videos and large image sequences.

## Features
- Clusters files using TF-IDF and DBSCAN
- Identifies sequences based on filename similarity
- Generates semantic names for sequences (AI-assisted if `sentence-transformers` is installed)
- Creates preview GIFs for videos and large image sequences
- Outputs a detailed JSON catalog with metadata, timing, and file information

## Requirements
- Python 3.7+
- Required packages:
  - numpy
  - scikit-learn
  - opencv-python
  - Pillow (for GIF creation)
  - sentence-transformers (optional, for improved sequence naming)

Install dependencies:
```bash
pip install numpy scikit-learn opencv-python Pillow sentence-transformers
```

## Usage
Run the clustering tool from the command line:
```bash
python cluster_sequence_file_cataloger.py --directory /path/to/your/files
```
Optional arguments:
- `--output <output.json>`: Specify output JSON file name

Example:
```bash
python cluster_sequence_file_cataloger.py --directory "/Volumes/omata/C4D Projects and Renders/Render Animations/Motion Track Studies" --output results.json
```

## Output
The tool generates a JSON file with the following structure:
- `metadata`: Scan info, timing, and summary statistics
- `sequences`: Detected file sequences with semantic names and file metadata
- `clusters`: Other file clusters
- `unclustered`: Files not grouped
- `videos`: Video files with preview GIFs and metadata

## Schema Summary
See below for a summary of the output JSON schema:
```typescript
interface FilesClusteringData {
  metadata: {
    root_path: string;
    scan_date: string;
    total_images: number;
    total_clusters: number;
    total_sequences: number;
    unclustered_count: number;
    timing: {
      total_seconds: number;
      file_discovery_seconds: number;
      directory_grouping_seconds: number;
      clustering_seconds: number;
      sequence_naming_seconds: number;
      gif_creation_seconds: number;
      json_preparation_seconds: number;
    }
  };
  sequences: Sequence[];
  clusters: Cluster[];
  unclustered: File[];
  videos?: Video[];
}

interface Sequence {
  id: string;
  name: string;
  directory: string;
  count: number;
  files: File[];
  preview_gif?: string;
}

interface Cluster {
  id: string;
  directory: string;
  count: number;
  files: File[];
}

interface File {
  filename: string;
  filepath: string;
  filesize: number;
  created: string;
  modified: string;
}

interface Video {
  id: string;
  name: string;
  directory: string;
  filepath: string;
  preview_gif: string;
  metadata: File;
}
```

## Debugging
You can use the provided VS Code launch configurations for debugging. See `.vscode/launch.json` for details.

## License
MIT License
