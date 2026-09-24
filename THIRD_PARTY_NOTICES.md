# Third-party notices

The Apache-2.0 license in `LICENSE` applies to the DCAG code except where an upstream file or directory specifies a different license. Original copyright headers are retained.

- **LLaVA**: https://github.com/haotian-liu/LLaVA. The `llava` integration derives from LLaVA and its FastChat/Alpaca training components. Apache-2.0; see source headers. Modified for domain-conditioned adapters, continual state management, and benchmark execution.
- **MCITlib**: https://github.com/Ghy0501/MCITlib. The continual training integration and MLLM-DCL evaluators are adapted from MCITlib. The evaluator answer-matching rules are retained.
- **MAE**: https://github.com/facebookresearch/mae. `domain_encoders/rs/model_mae_af.py` derives from MAE, Copyright (c) Meta Platforms, Inc. and affiliates. It is adapted to expose intermediate remote-sensing features and is covered by **CC BY-NC 4.0**, not the root Apache-2.0 license. See `domain_encoders/rs/LICENSE` and https://creativecommons.org/licenses/by-nc/4.0/legalcode.
- The GeoMIM, Pix2Struct, PubMedCLIP, and CandleFusion integrations load externally supplied checkpoints. Their weights and upstream projects are not relicensed by this repository.

The architecture illustration accompanies the DCAG paper. Pretrained checkpoints and benchmark datasets are not distributed in this code package.
