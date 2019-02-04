#!/bin/bash
# delete DSStore
find . -name '.DS_Store' -type f -delete
# delete direcotry
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
