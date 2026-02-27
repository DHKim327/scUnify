# Registry — Model Modules Attribution

The `models/modules/` directory contains model architecture code adapted from the original implementations of each foundation model. These files have been minimally modified to integrate with scUnify's unified inference pipeline.

## Source Attribution

| Model | Module Path | Original Repository | License |
|---|---|---|---|
| scFoundation | `modules/scfoundation/` | [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation) | MIT |
| scGPT | `modules/scgpt/` | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | BSD-3-Clause |
| UCE | `modules/uce/` | [snap-stanford/UCE](https://github.com/snap-stanford/UCE) | MIT |

## Structure

```
registry/
├── dataset/                    # scUnify dataset/tokenizer wrappers
│   ├── _scfoundation_dataset.py
│   ├── _scgpt_dataset.py
│   └── _uce_dataset.py
├── models/
│   ├── _scfoundation_wrapper.py  # scUnify inferencer wrapper
│   ├── _scgpt_wrapper.py
│   ├── _uce_wrapper.py
│   └── modules/                  # Original model architectures (adapted)
│       ├── scfoundation/
│       ├── scgpt/
│       └── uce/
└── __init__.py
```

## Notes

- **`models/modules/`**: Contains model architecture code derived from the original repositories listed above. Modifications are limited to interface compatibility with scUnify (e.g., import paths, forward signature adjustments).
- **`models/_*_wrapper.py`**: scUnify-specific wrapper classes that implement the standardized inferencer interface (`preprocessing`, `load_model`, `forward`).
- **`dataset/_*_dataset.py`**: scUnify-specific dataset/tokenizer implementations for each model.

All credit for the original model architectures belongs to their respective authors.
