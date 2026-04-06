#!/bin/bash

# Exit immediately if a command fails
set -e

TARGET_DIR="./outputs/MelTransformerFrame/"

# Check if directory exists
if [ -d "$TARGET_DIR" ]; then
    echo "Removing all files in $TARGET_DIR..."

    # Remove all files (but keep the directory itself)
    rm -f "$TARGET_DIR"/*

    echo "Done."
else
    echo "Directory $TARGET_DIR does not exist."
fi

TARGET_DIR="./outputs/MelTransformerFrameBin"

# Check if directory exists
if [ -d "$TARGET_DIR" ]; then
    echo "Removing all files in $TARGET_DIR..."

    # Remove all files (but keep the directory itself)
    rm -f "$TARGET_DIR"/*

    echo "Done."
else
    echo "Directory $TARGET_DIR does not exist."
fi