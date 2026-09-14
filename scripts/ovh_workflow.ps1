###############################################################################
# SELMA Reading Policy — OVHcloud AI Training Workflow
# All values filled in from verified CLI output.
# Run each section ONE AT A TIME and confirm before proceeding.
###############################################################################

# ─────────────────────────────────────────────────────────────────────────────
# Confirmed values (do not change unless you recreate the resources)
# ─────────────────────────────────────────────────────────────────────────────
$OVH_REGION       = "GRA"
$OVH_REGISTRY     = "registry.gra.ai.cloud.ovh.net/9d2529e9-39da-42e0-8004-25a4c9731350"
$IMAGE_TAG        = "$OVH_REGISTRY/selma-train:latest"

$CONTAINER_MODEL  = "selma-model"
$CONTAINER_DATA   = "selma-data"
$CONTAINER_OUTPUT = "selma-output"

# ─────────────────────────────────────────────────────────────────────────────
# STEPS 1-4  DONE
# ─────────────────────────────────────────────────────────────────────────────
# [x] Step 1  GSM8K dataset prepared (data/gsm8k/train.parquet, test.parquet)
# [x] Step 2  Qwen2-7B-Instruct downloaded (models/Qwen2-7B-Instruct, 14.1 GB)
# [x] Step 3  Buckets created: selma-model, selma-data, selma-output
# [x] Step 4  Data uploaded:
#               selma-data@GRA  →  data/gsm8k/train.parquet
#                                  data/gsm8k/test.parquet
#               selma-model@GRA →  models/Qwen2-7B-Instruct/... (14.1 GB)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  — Build the Docker image   (run from PROJECT ROOT in Windows terminal)
# ─────────────────────────────────────────────────────────────────────────────
# >>> docker build -f docker/Dockerfile -t selma-train:latest .
#
# First build: 25-40 min (flash-attn compilation).
# Watch for "Successfully built <id>" at the end.
# If the build errors, share the last 20 lines of output.

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  — Push the image to OVHcloud Container Registry
# ─────────────────────────────────────────────────────────────────────────────
# The registry URL is the SHARED registry for your project.
# Login credentials: your OVHcloud account username + password
# (or an application token created in the OVH Manager).
#
# >>> docker login registry.gra.ai.cloud.ovh.net
#
# Tag and push:
# >>> docker tag selma-train:latest $IMAGE_TAG
# >>> docker push $IMAGE_TAG
#
# Expected: "latest: digest: sha256:..." on success.

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  — Verify the flavor name
# ─────────────────────────────────────────────────────────────────────────────
# >>> ovhai capabilities flavor list
#
# Look for the L40s entry (name contains "l40s").
# Common names: "ai1-1-gpu-l40s"  or  "l40s"
# Set it below after confirming:
$FLAVOR = "l40s-1-gpu"   # confirmed from ovhai capabilities flavor list

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  — Submit the training job
# ─────────────────────────────────────────────────────────────────────────────
# Volume mount strategy:
#   selma-model@GRA  with prefix  models/Qwen2-7B-Instruct  → /workspace/model
#   selma-data@GRA   with prefix  data/gsm8k                → /workspace/data
#   selma-output@GRA (no prefix)                            → /workspace/output  (rw)
#
# Inside the job the paths will be:
#   /workspace/model/config.json                  (model root)
#   /workspace/data/train.parquet
#   /workspace/data/test.parquet
#   /workspace/output/checkpoints/               (written by trainer)

# Print the command first, then uncomment the bottom line to actually submit.
$JOB_SUBMIT = @"
ovhai job run `
  --name selma-reading-policy `
  --gpu 1 `
  --flavor $FLAVOR `
  -v "selma-model@GRA/models/Qwen2-7B-Instruct:/workspace/model:ro" `
  -v "selma-data@GRA/data/gsm8k:/workspace/data:ro" `
  -v "selma-output@GRA:/workspace/output:rw" `
  $IMAGE_TAG `
  -- `
  --model_name /workspace/model `
  --train_file /workspace/data/train.parquet `
  --val_file   /workspace/data/test.parquet `
  --task gsm8k `
  --n_rollouts 8 `
  --batch_size 8 `
  --max_steps 500 `
  --output_dir /workspace/output/checkpoints `
  --device cuda `
  --log_freq 10 `
  --eval_freq 50 `
  --save_freq 50
"@

Write-Host "=== Job submission command ==="
Write-Host $JOB_SUBMIT
# Uncomment to submit:
# Invoke-Expression $JOB_SUBMIT

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9  — Monitor the job  (run in Ubuntu / ovhai terminal)
# ─────────────────────────────────────────────────────────────────────────────
# List jobs:
#   ovhai job list
#
# Stream live logs (replace JOB_ID with the ID returned by job run):
#   ovhai job logs <JOB_ID> --follow
#
# Check job status:
#   ovhai job get <JOB_ID>
#
# Download checkpoints once training finishes:
#   ovhai bucket object download selma-output@GRA --prefix checkpoints/ --output ./checkpoints_from_ovh/
