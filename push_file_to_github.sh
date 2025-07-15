#!/bin/bash

# Usage check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file-to-upload>"
    exit 1
fi

FILE=$1

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found!"
    exit 2
fi

# Ensure this is a Git repo
if [ ! -d .git ]; then
    echo "Error: This directory is not a Git repository."
    exit 3
fi

# Stage, commit, and push
git add "$FILE"
git commit -m "Add/update $FILE"
git push origin $(git rev-parse --abbrev-ref HEAD)

