#!/bin/bash
# Project Phoenix: V4 Clean Slate Protocol initialization
# Role: Lead Systems Architect

# PHASE 1: The Purge & Archive
echo "[*] Initiating V4 Purge..."

mkdir -p archive_v3/

# Safe move of telemetry and logs
if [ -f "training_stats.json" ] || [ -f "training.log" ] || ls *.bak >/dev/null 2>&1; then
    echo "[+] Archiving V3 telemetry artifacts..."
    mv training_stats.json training.log *.bak archive_v3/ 2>/dev/null
else
    echo "[-] No telemetry files found to archive."
fi

# Hard purge of checkpoints
echo "[!] Deleting V3 weight modules (dim_llm_*.pt)..."
rm -f dim_llm_*.pt

echo "[✓] V4 Environment Ready. Proceed to PHASE 2."
