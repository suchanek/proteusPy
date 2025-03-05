#!/usr/bin/env python3
"""
Script to find machine learning model files on a MacOS system.
Searches for common ML model file extensions and reports their locations and sizes.
"""

import argparse
import os
import pathlib
import sys
from collections import defaultdict
from datetime import datetime

# Common ML model file extensions
ML_MODEL_EXTENSIONS = [
    # TensorFlow
    ".pb",
    ".tflite",
    ".savedmodel",
    ".tf",
    ".index",
    ".meta",
    ".data-*",
    # PyTorch
    ".pt",
    ".pth",
    ".ckpt",
    # Keras
    ".h5",
    ".keras",
    ".hdf5",
    # ONNX
    ".onnx",
    # Scikit-learn
    ".pkl",
    ".joblib",
    ".pickle",
    # XGBoost
    ".model",
    ".xgb",
    # Other
    ".bin",
    ".weights",
    ".mlmodel",
    ".caffemodel",
    ".params",
    ".gguf",
]


def get_size_str(size_bytes):
    """Convert size in bytes to a human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def find_ml_models(
    start_dir, exclude_dirs=None, max_depth=None, min_size=0, verbose=False
):
    """
    Find ML model files recursively starting from start_dir

    Args:
        start_dir (str): Directory to start the search from
        exclude_dirs (list): List of directory names to exclude
        max_depth (int): Maximum directory depth to search
        min_size (int): Minimum file size in bytes to include

    Returns:
        list: List of tuples (file_path, file_size)
    """
    if exclude_dirs is None:
        exclude_dirs = [
            ".git",
            "node_modules",
            "venv",
            "env",
            ".venv",
            ".env",
            "__pycache__",
        ]

    ml_files = []
    start_path = pathlib.Path(start_dir).expanduser().resolve()

    print(f"Starting search from: {start_path}")
    print("This may take some time depending on the number of files...")

    # Track statistics
    stats = {"dirs_searched": 0, "files_checked": 0, "models_found": 0, "errors": 0}

    # Group models by type
    models_by_type = defaultdict(list)

    # For progress tracking
    last_dir_reported = ""
    progress_update_interval = 1000  # Update directory progress every N directories

    try:
        for root, dirs, files in os.walk(start_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            # Check depth limit
            if max_depth is not None:
                rel_path = os.path.relpath(root, start_path)
                depth = len(rel_path.split(os.sep)) if rel_path != "." else 0
                if depth > max_depth:
                    dirs[:] = []  # Don't go deeper
                    continue

            stats["dirs_searched"] += 1

            # Show current directory being searched (periodically to avoid console spam)
            if stats["dirs_searched"] % progress_update_interval == 0 and verbose:
                rel_path = os.path.relpath(root, start_path)
                if rel_path != last_dir_reported and verbose:
                    print(f"Searching directory: {rel_path}")
                    last_dir_reported = rel_path

            # Process files in this directory
            for filename in files:
                stats["files_checked"] += 1

                # Check if the file has an ML model extension
                is_model = False
                for ext in ML_MODEL_EXTENSIONS:
                    if ext.endswith("*"):
                        # Handle patterns like .data-*
                        base_ext = ext[:-1]
                        if filename.endswith(base_ext) or any(
                            filename.endswith(f"{base_ext}{i}") for i in range(10)
                        ):
                            is_model = True
                            break
                    elif filename.endswith(ext):
                        is_model = True
                        break

                if is_model:
                    try:
                        file_path = os.path.join(root, filename)
                        file_size = os.path.getsize(file_path)

                        # Skip if smaller than minimum size
                        if file_size < min_size:
                            continue

                        # Get file extension
                        _, ext = os.path.splitext(filename.lower())
                        if not ext and "." in filename:
                            # Handle complex extensions like .tar.gz
                            parts = filename.split(".")
                            if len(parts) > 1:
                                ext = f".{parts[-1]}"

                        # Add to results
                        ml_files.append((file_path, file_size))
                        models_by_type[ext].append((file_path, file_size))
                        stats["models_found"] += 1

                        # Print progress for large searches
                        if stats["models_found"] % 100 == 0:
                            print(
                                f"Found {stats['models_found']} model files so far..."
                            )
                            # Also show current directory when reporting model count
                            rel_path = os.path.relpath(root, start_path)
                            if verbose:
                                print(f"Current directory: {rel_path}")

                    except (PermissionError, OSError) as e:
                        stats["errors"] += 1
                        print(
                            f"Error accessing {os.path.join(root, filename)}: {e}",
                            file=sys.stderr,
                        )

    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    return ml_files, stats, models_by_type


def main():
    parser = argparse.ArgumentParser(
        description="Find machine learning model files on your system"
    )
    parser.add_argument(
        "--start-dir",
        "-d",
        default="~",
        help="Directory to start searching from (default: home directory)",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        nargs="+",
        default=None,
        help="Directories to exclude from search",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file to save results (default: print to console)",
    )
    parser.add_argument(
        "--max-depth",
        "-m",
        type=int,
        default=None,
        help="Maximum directory depth to search",
    )
    parser.add_argument(
        "--min-size",
        "-s",
        type=float,
        default=0,
        help="Minimum file size in MB to include",
    )
    parser.add_argument(
        "--sort",
        choices=["size", "path"],
        default="size",
        help="Sort results by size (largest first) or path (default: size)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show more detailed progress information",
    )

    args = parser.parse_args()

    # Convert min_size from MB to bytes
    min_size_bytes = int(args.min_size * 1024 * 1024)

    start_time = datetime.now()
    print(f"Search started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    ml_files, stats, models_by_type = find_ml_models(
        args.start_dir,
        exclude_dirs=args.exclude,
        max_depth=args.max_depth,
        min_size=min_size_bytes,
        verbose=args.verbose,
    )

    # Sort results
    if args.sort == "size":
        ml_files.sort(key=lambda x: x[1], reverse=True)
    else:  # sort by path
        ml_files.sort(key=lambda x: x[0])

    # Prepare output
    output_lines = [
        f"ML Model Files Search Results",
        f"===========================",
        f"Search started from: {args.start_dir}",
        f"Search completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {datetime.now() - start_time}",
        f"",
        f"Statistics:",
        f"  Directories searched: {stats['dirs_searched']}",
        f"  Files checked: {stats['files_checked']}",
        f"  Model files found: {stats['models_found']}",
        f"  Errors encountered: {stats['errors']}",
        f"",
        f"Results:",
        f"--------",
    ]

    # Add file results
    if ml_files:
        for file_path, file_size in ml_files:
            output_lines.append(f"{file_path} ({get_size_str(file_size)})")
    else:
        output_lines.append("No ML model files found.")

    # Add summary by file type
    output_lines.extend([f"", f"Summary by File Type:", f"--------------------"])

    for ext, files in sorted(
        models_by_type.items(), key=lambda x: sum(f[1] for f in x[1]), reverse=True
    ):
        total_size = sum(f[1] for f in files)
        output_lines.append(
            f"{ext}: {len(files)} files, total size: {get_size_str(total_size)}"
        )

    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(f"{line}\n")
        print(f"Results saved to {args.output}")
    else:
        for line in output_lines:
            print(line)


if __name__ == "__main__":
    main()
