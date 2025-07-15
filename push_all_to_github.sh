#!/bin/bash

# Exit on error
set -e

# ==== CONFIGURABLE ====
github_repo="https://github.com/arjunsingh4545/satellite_image_change_detection.git"
# =======================

# Step 1: Make sure we’re in a Git repo
if [ ! -d .git ]; then
    echo "❌ Not a Git repository. Run 'git init' first."
    exit 1
fi

# Step 2: Set or verify remote origin
current_origin=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$current_origin" ]; then
    echo "🔗 Setting remote origin to $github_repo"
    git remote add origin "$github_repo"
elif [ "$current_origin" != "$github_repo" ]; then
    echo "⚠️  Remote origin mismatch."
    echo "Current: $current_origin"
    echo "Expected: $github_repo"
    echo "Run: git remote set-url origin \"$github_repo\" to fix."
    exit 2
fi

# Step 3: Track large files using Git LFS (files > 50MB)
find . -type f -size +50M | while read -r file; do
    git lfs track "$file"
done

# Step 4: Add .gitattributes if LFS was updated
git add .gitattributes

# Step 5: Add, commit, and push all changes
git add .
git commit -m "Upload all files with LFS support (auto-tracked large files)"

# Step 6: Push to current branch
branch=$(git rev-parse --abbrev-ref HEAD)
git push origin "$branch"

