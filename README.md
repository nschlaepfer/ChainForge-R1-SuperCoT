# ChainForge

ChainForge demonstrates a multi-stage workflow for training Qwen2.5 models using
ideas from the [DeepSeek-R1 paper](https://arxiv.org/pdf/2501.12948). It combines
DeepSeek Reasoner's chain-of-thought (CoT) output with optional Anthropic Claude
expansions. This revision includes a unified `call_model()` helper and adds
support for Claude 4, the latest DeepSeek-R1 checkpoint, and Mistral's Devstral
model.

Key stages include:

1. **Hybrid CoT Collection** – gather reasoning traces from DeepSeek and expand uncertain steps with Claude.
2. **Cold-Start SFT** – fine-tune on the collected CoT data.
3. **Reasoning-Oriented RL** – train the model with a GRPO-style algorithm.
4. **Rejection Sampling** – filter the best RL completions and run an additional SFT pass.
5. **Final RL & Optional Distillation** – further improve the model and optionally distill to smaller checkpoints.
6. A placeholder `diffusion_refine()` hook exists for future Diffusion-of-Thought reasoning refinements.

## Requirements

- Python 3.8+
- A GPU is recommended for RL stages
- `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY` environment variables
- `pip install -r requirements.txt`

## Usage

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="..."
export ANTHROPIC_API_KEY="..."
python deepseek_qwen2_5_integration_r1.py
```

## Project Structure

```
.
├── deepseek_qwen2_5_integration_r1.py  # Main pipeline
├── requirements.txt
└── README.md
```

## MLX-GRPO Enhancements

The training script now includes features inspired by the MLX-GRPO project:

- **Dataclass configs** – `TrainingArgs` and `RewardConfig` centralise
  hyper-parameters and reward weights.
- **Modular rewards** – format and content rewards can be combined for
  verifiable tasks.
- **Adaptive KL penalty** and **atomic checkpointing** ensure stable RL runs
  that can be resumed from the last checkpoint.
- Optional **KV cache quantisation** and basic **speculative decoding** speed
  up generation on Apple Silicon.

## Citation

If you use this project, please cite the DeepSeek-R1 paper:

```bibtex
@misc{deepseek2024r1,
  title={DeepSeek-R1: Augmenting Reasoning via Reinforcement Learning},
  author={DeepSeek Team},
  year={2024},
  publisher={arXiv}
}
```

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
