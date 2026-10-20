# Environment and dataset setup

Run commands from the repository root.

## Environment

Requires Linux, an NVIDIA GPU, and the CUDA 12.8 Toolkit.

```bash
conda env create -f environment.yaml
conda activate wovenanchor
python -m pip install --no-build-isolation -r requirements-cuda.txt
```

[environment.yaml](../environment.yaml) installs PyTorch and the other dependencies;
[requirements-cuda.txt](../requirements-cuda.txt) installs the native extensions afterward.

## MegaDepth

Follow the [LoFTR guide](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md)
to download depth maps, D2-Net undistorted images, and training/validation indices:

```bash
mkdir -p data/megadepth/train data/megadepth/test
ln -s /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM data/megadepth/train/
ln -s /path/to/megadepth/phoenix /path/to/megadepth_d2net/Undistorted_SfM data/megadepth/test/
ln -s /path/to/megadepth_indices data/megadepth/index
```

Training and validation both require depth maps.

For evaluation only, download the MegaDepth-1500 test data from the same guide.

## ScanNet-1500

Prepare the ScanNet-1500 test subset using the
[LoFTR guide](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md),
following [ScanNet's access terms](https://github.com/ScanNet/ScanNet#scannet-data):

```bash
mkdir -p data/scannet
ln -s /path/to/scannet-1500-testset data/scannet/test
```

## HPatches

Download the [HPatches full sequences](https://github.com/hpatches/hpatches-dataset#full-image-sequences):

```bash
mkdir -p data/hpatches
ln -s /path/to/hpatches-sequences-release data/hpatches/hpatches-sequences-release
```

We evaluate 108 scenes (540 image pairs); the loader automatically excludes eight scenes.

See the [README](../README.md#training-and-evaluation) for training, evaluation,
and checkpoint instructions.
