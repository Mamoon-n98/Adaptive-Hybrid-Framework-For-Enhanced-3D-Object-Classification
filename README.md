# Hybrid Model Project

## Overview
This project implements a state-of-the-art hybrid 3D point cloud model combining dynamic voxelization, hierarchical PointNet, sparse VoxelNet, feature fusion, and modified BLS for high accuracy (>93%) on ModelNet10/40 and KITTI.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download datasets to `ModelNet40` and `KITTI` folders.
3. Run training: `./run.sh train`
4. Evaluate: `./run.sh eval`

## Features
- Dynamic voxelization for efficiency.
- Hybrid integration for robustness.
- Enhanced BLS for fast training.
- Scalable for real-time apps.
- Robust to noise and incompleteness.