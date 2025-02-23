#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Ensure eaton_db directory exists and has correct permissions
mkdir -p eaton_db
chmod -R 755 eaton_db
