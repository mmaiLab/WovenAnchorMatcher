# WovenAnchor Matcher (WAM)

## Setup

Follow the [installation and dataset setup guide](docs/SETUP.md).

## Training and evaluation

After preparing the environment and datasets, train and evaluate with:

```bash
# Train on MegaDepth.
bash scripts/reproduce_train/outdoor.sh

# Evaluate on MegaDepth-1500, ScanNet-1500, or HPatches.
bash scripts/reproduce_test/outdoor.sh megadepth
bash scripts/reproduce_test/outdoor.sh scannet
bash scripts/reproduce_test/outdoor.sh hpatches
```

Training uses batch size 8 and 20 epochs, saving checkpoints under `logs/`.
Logging uses TensorBoard by default; add `--logger wandb` to use W&B.
Other options are listed by `python train.py --help` and `python test.py --help`.

## Checkpoints

Download the pretrained checkpoint from GitHub Releases:

```bash
mkdir -p weights
wget -O weights/outdoor.ckpt \
  "https://github.com/mmaiLab/WovenAnchorMatcher/releases/download/v1.0/outdoor.ckpt"
```

Evaluation uses `weights/outdoor.ckpt` by default; add
`--ckpt_path /path/to/model.ckpt` to use another checkpoint.

When no model checkpoint is provided, backbone initialization is downloaded automatically from the
[JamMa release](https://github.com/leoluxxx/JamMa/releases/download/v0.1/convnextv2_nano_pretrain.ckpt).

## Ablations

Pass a configuration from [configs/woven_anchor/ablations/](configs/woven_anchor/ablations/)
to both training and evaluation:

```bash
bash scripts/reproduce_train/outdoor.sh \
  --exp_name inter_crossbimamba \
  --ablation_cfg configs/woven_anchor/ablations/inter_crossbimamba.py

bash scripts/reproduce_test/outdoor.sh megadepth \
  --ablation_cfg configs/woven_anchor/ablations/inter_crossbimamba.py \
  --ckpt_path /path/to/inter_crossbimamba.ckpt
```

## Visualization and statistics

Select a visualization with `--vis`:

| Mode | Description | Output directory |
| --- | --- | --- |
| `qualitative` | Matches | `visualization/qualitative/` |
| `anchor` | Assignment maps | `visualization/anchor_probs/` |
| `woven` | Effective receptive fields | `visualization/woven_erf/` |

`--vis_max_pairs` defaults to 10; use `-1` for all pairs or `0` to disable
visualization. This limits figures only; evaluation still processes all pairs.

```bash
bash scripts/reproduce_test/outdoor.sh megadepth --vis anchor --vis_max_pairs 5
bash scripts/reproduce_test/outdoor.sh megadepth --stats_anchor
```

Anchor statistics are aggregated over all evaluation pairs and printed to the log.
`--vis woven` saves PNG figures and raw, unnormalized ERF maps in NPZ files.
Measure latency in a separate run without visualization.

For single-scene MegaDepth visualization (300 pairs per scene):

```bash
python test.py configs/data/megadepth_test_0015.py configs/woven_anchor/outdoor/eval.py \
  --ckpt_path /path/to/model.ckpt --vis anchor --vis_name 0015 --vis_max_pairs 10
```

Use `configs/data/megadepth_test_0022.py` with `--vis_name 0022` for scene 0022.

## License

Original code contributions are licensed under [MIT](LICENSE); inherited components
retain their respective licenses. The pretrained checkpoint is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

The framework builds on
JamMa and LoFTR, with components from Mamba, ConvNeXt V2, and GlueFactory.
See [third-party notices](THIRD_PARTY_NOTICES.md) for code licenses, credits,
and dataset provenance and terms.
