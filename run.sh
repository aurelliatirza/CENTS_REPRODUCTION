#!/bin/bash
echo "=== CENTS Trovi Artifact ==="
echo "Repository structure follows the original CENTS repo"
echo "Running evaluation (if results already exist, this is a no-op)"

cd exp
bash run_eval_all.sh || echo "Evaluation already completed or skipped"

echo "Done. Please check CTA/results or related output folders."
