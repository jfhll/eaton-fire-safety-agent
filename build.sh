#!/bin/bash
# Unzip eaton_db.zip if it exists (assume it’s manually uploaded or included in Render)
if [ -f eaton_db.zip ]; then
    unzip -o eaton_db.zip || echo "eaton_db.zip already extracted or not found"
fi
