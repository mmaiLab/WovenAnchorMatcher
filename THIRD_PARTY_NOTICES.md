# Third-party notices

The root [MIT license](LICENSE) applies to original WovenAnchor contributions.
It does not replace the licenses of inherited code, dataset assets, or model
weights. Existing copyright and license notices remain applicable.

## JamMa and the LoFTR framework

WovenAnchor is adapted from [JamMa](https://github.com/leoluxxx/JamMa),
copyright (c) 2025 leo, under the [MIT license](LICENSES/JamMa-MIT.txt).
Inherited components include the training/evaluation infrastructure, dataset
loaders, optimizer utilities, geometric supervision, and coarse-to-fine matching
framework. WovenAnchor changes the context architecture, matching implementation,
configuration, evaluation helpers, and public interfaces.

JamMa acknowledges [LoFTR](https://github.com/zju3dv/LoFTR) as an upstream
implementation. The [LoFTR license](https://github.com/zju3dv/LoFTR/blob/master/LICENSE)
is Apache-2.0; a complete copy is included as
[Apache-2.0.txt](LICENSES/Apache-2.0.txt).
Those terms continue to apply to its inherited portions, including dataset,
training, matching, and geometry utilities. Do not treat the MIT license of
JamMa or WovenAnchor as replacing the upstream Apache-2.0 terms.

The original upstream revisions of the inherited code were not recorded.
Current upstream license files establish the recorded license text, not the
revision of each copied source file. See the cited projects and their papers
for the original implementations and authors.

## Mamba

[Mamba](https://github.com/state-spaces/mamba/tree/v2.2.5), copyright (c) 2023
Tri Dao and Albert Gu, is distributed under
[Apache-2.0](LICENSES/Apache-2.0.txt). The environment pins `mamba-ssm==2.2.5`.
WovenMamba adapts the selective-SSM parameterization and time-step initialization
from `mamba_ssm/modules/mamba_simple.py`; ablation initialization follows
`mamba_ssm/models/mixer_seq_simple.py`. WovenAnchor adds spatial convolutions,
horizontal/vertical composition, separator handling, masking, and its block API.
The selective scan kernel, Mamba ablation module, and RMSNorm kernels are imported
from the installed package, not vendored here.

## Detectron2

`src/utils/comm.py` was copied from
[Detectron2](https://github.com/facebookresearch/detectron2) through the inherited
framework. Its copyright notice is retained:
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
Detectron2 is licensed under [Apache-2.0](LICENSES/Apache-2.0.txt).

## GlueFactory and Sacred

The selected GlueFactory files and the modified Camera/Pose utilities are covered
by their [LICENSE](src/third_party/gluefactory/LICENSE) and
[provenance notice](src/third_party/gluefactory/NOTICE).

`src/third_party/gluefactory/utils/stdout_capturing.py` also derives from
[Sacred](https://github.com/IDSIA/sacred), copyright (c) 2014 Klaus Greff,
under the [MIT license](LICENSES/Sacred-MIT.txt). The file retains its attribution
to Paul-Edouard Sarlin for the adaptation.

## ConvNeXt V2 and pretrained weights

The adapted ConvNeXt V2 source retains the Meta copyright notice and is covered
by its [LICENSE](src/third_party/convnextv2/LICENSE) and
[provenance notice](src/third_party/convnextv2/NOTICE).
The official code is MIT; official pretrained/fine-tuned weights carry
CC-BY-NC 4.0 terms. WovenAnchor obtains its backbone initialization from JamMa's
`convnextv2_nano_pretrain.ckpt` release asset. Downloading the asset from JamMa
does not replace applicable upstream weight terms with the JamMa code license.
No third-party checkpoint is bundled in this source package. The root MIT
license is not a license grant for pretrained or derived model weights.

## WovenAnchor checkpoint

The pretrained WovenAnchor checkpoint (`outdoor.ckpt`) is released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Dataset assets

`assets/` contains evaluation pair lists and metadata inherited through
[JamMa](https://github.com/leoluxxx/JamMa) and
[LoFTR](https://github.com/zju3dv/LoFTR). Images and depth maps are obtained
separately; see the [setup guide](docs/SETUP.md). Dataset assets retain their
original terms and are not relicensed under the project's MIT license.

- [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/): released derived
  data uses CC BY 4.0; original photographs retain their own rights.
  Cite Zhengqi Li and Noah Snavely (CVPR 2018).
- [ScanNet](https://github.com/ScanNet/ScanNet): use requires agreement to the
  [Terms of Use](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf), including
  non-commercial research/education and access-sharing conditions.
  Cite Angela Dai et al. (CVPR 2017).

This document is WovenAnchor's attribution record, not an upstream NOTICE file.
