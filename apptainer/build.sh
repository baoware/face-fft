#!/usr/bin/env bash
# build.sh

set -euo pipefail

# Build the Apptainer image from the definition
echo "Building face_fft.sif from face_fft.def..."
apptainer build face_fft.sif face_fft.def

# Deploy to Rivanna (requires VPN or on-campus network)
echo "Copying face_fft.sif to Rivanna project directory..."
scp -C face_fft.sif "$REMOTE"

echo "Build and deploy complete."
