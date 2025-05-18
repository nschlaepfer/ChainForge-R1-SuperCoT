# R1 + Super CoT: ChainForge

![ChainForge](https://github.com/user-attachments/assets/5b6d4341-f632-41d5-9117-a82a9a4b983a)

A multi-stage pipeline that integrates **DeepSeek Reasoner** and **Anthropic Claude** for enhanced chain-of-thought data generation, built on **Qwen2.5** language models. Inspired by the [DeepSeek-R1 paper](https://arxiv.org/pdf/2501.12948), it showcases:

1. **Hybrid CoT Generation** using DeepSeek + Anthropic expansions
2. **Cold-Start SFT** using enhanced CoT data
3. **Reasoning-Oriented RL** (GRPO) for improved correctness
4. **Rejection Sampling** to gather top responses
5. **Additional SFT** on filtered data
6. **Final RL** for broad scenarios
7. **Optional Distillation** to smaller Qwen2.5 checkpoints

## What is Super Chain of Thought?

Super Chain of Thought (Super CoT) is an enhanced reasoning framework that combines DeepSeek's chain-of-thought capabilities with selective Anthropic expansions and reinforcement learning. Unlike traditional CoT which simply shows reasoning steps, Super CoT:

1. **Structured Reasoning**: Uses DeepSeek's <think> tags with Anthropic expansions for uncertain steps
2. **Iterative Refinement**: Applies RL to improve reasoning quality over multiple stages
3. **Quality Control**: Implements rejection sampling to filter and keep only the best reasoning paths
4. **Knowledge Distillation**: Transfers learned reasoning patterns to smaller, more efficient models

## Anthropic Integration Details

The integration of Anthropic Claude adds a crucial layer of reasoning enhancement to our pipeline:

### 1. Uncertainty Detection

* **Automatic Detection**: Scans reasoning steps for uncertainty markers like:
  * "maybe", "not sure", "guess", "uncertain", "unsure"
  * Length-based heuristics for complex steps
  * Domain-specific uncertainty signals
* **Selective Expansion**: Only expands steps that need clarification
* **Preservation of Clear Reasoning**: Leaves well-reasoned steps untouched

### 2. Expansion Process

1. **Input Processing**:
   ```python
   # Original DeepSeek step
   <think>This might be related to quantum tunneling, but I'm not sure...</think>

   # Anthropic expansion request
   "Please provide a factual grounding of why this step might be correct..."

   # Final expanded format
   <think>Original step
   <explanation>Anthropic's detailed grounding</explanation>
   </think>
   ```

2. **Integration Points**:
   * During initial CoT collection
   * In rejection sampling phase
   * During final model distillation

### 3. Implementation Details

1. **API Integration**:
   ```python
   # Setup
   anthropic_client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])

   # Expansion call
   expansion = anthropic_client.completions.create(
       model="claude-3-5-sonnet-20241022",
       max_tokens=512,
       prompt=f"Explain why this step is valid: {uncertain_step}"
   )
   ```

2. **Error Handling**:
   * Graceful fallbacks to original reasoning
   * Rate limiting protection
   * Context length management
   * Expansion validation

3. **Best Practices**:
   * Keep expansions concise (≤512 tokens)
   * Focus on factual grounding
   * Maintain reasoning flow
   * Preserve original insights

## Latest Results & Achievements

Based on the DeepSeek-R1 paper:

1. **Math & Reasoning**:
   * 79.8% Pass@1 on AIME 2024 (surpassing OpenAI-o1-1217)
   * 97.3% on MATH-500 (on par with OpenAI-o1-1217)
   * Strong performance on MMLU, MMLU-Pro, and GPQA Diamond

2. **Coding**:
   * 2,029 Elo rating on Codeforces (96.3 percentile)
   * Strong performance on LiveCodeBench
   * Competitive results on software engineering tasks

3. **Distilled Models**:
   * DeepSeek-R1-Distill-Qwen-7B: 55.5% on AIME 2024
   * DeepSeek-R1-Distill-Qwen-32B: 72.6% on AIME 2024, 94.3% on MATH-500

## Key Features & Updates

1. **Hybrid CoT Generation**:
   * DeepSeek Reasoner for base chain-of-thought
   * Anthropic Claude for expanding "uncertain" steps
   * Clean conversation history management
   * Automatic uncertainty detection

2. **Enhanced GRPO Implementation**:
   * Group-based advantage computation
   * KL-constrained optimization
   * Reference model comparison
   * Stable policy updates

3. **Prompting Best Practices**:
   * Zero-shot prompting recommended
   * Direct problem description preferred
   * Avoid few-shot examples (can degrade performance)
   * Clear output format specification

4. **Known Limitations**:
   * Language Mixing: Optimized for Chinese/English
   * Prompt Sensitivity: Performance varies with prompt structure
   * Software Engineering: Limited RL application due to evaluation time
   * Function Calling: May need additional fine-tuning for specific formats

## Paper Methodology & Our Implementation

This repository provides a Qwen2.5-based implementation inspired by the DeepSeek-R1 paper, enhanced with Anthropic expansions. We focus on making the core ideas accessible through a single, well-documented Python script. Here's how we implement the key concepts:

### 1. Base Architecture

* Uses **Qwen2.5-7B** as the foundation model
* Implements **GRPO** (Group Relative Policy Optimization)
* Integrates Anthropic Claude for uncertain step expansions
* Efficient training without separate critic models

### 2. Training Pipeline Stages

#### Stage 0: Hybrid CoT Generation

The pipeline begins by gathering high-quality chain-of-thought data from DeepSeek Reasoner and selectively expanding uncertain steps with Anthropic Claude:

1. **Response Format**:
   ```
   Question: {prompt}
   <reasoning_process>
     <think>Step-by-step logical deduction</think>
     <explanation>Anthropic expansion for uncertain steps</explanation>
   </reasoning_process>
   <summary>
     Final concise answer
   </summary>
   ```

2. **API Integration**:
   * DeepSeek Reasoner for base CoT
   * Anthropic Claude for uncertain step expansion
   * Clean conversation history
   * Automatic uncertainty detection

3. **Error Handling**:
   * API failures trigger fallbacks
   * Rate limiting protection
   * Response validation
   * Expansion integration checks

#### Stage 1: Cold-Start SFT

Initial supervised fine-tuning on enhanced CoT data:

1. **Data Processing**:
   * Tokenization with proper padding
   * Sequence length management
   * Batch collation
   * Expansion preservation

2. **Training Loop**:
   * Linear learning rate warmup
   * Gradient clipping
   * Progress tracking
   * Validation of reasoning structure

#### Stage 2: Reasoning-Oriented RL

Group-based Reward Policy Optimization (GRPO):

1. **Policy Architecture**:
   * Language model as base policy
   * Token-level probability computation
   * Group advantage estimation
   * KL divergence constraints

2. **Reward Structure**:
   * +1.0 for correct answers
   * +0.2 for proper reasoning format
   * Bonus for utilizing expansions
   * Normalized advantages within groups

#### Stage 3: Rejection Sampling & Additional SFT

Quality-focused data augmentation:

1. **Sampling Strategy**:
   * Multiple candidates per question
   * Temperature-controlled generation
   * Reward-based filtering
   * Expansion preservation check

2. **Additional Training**:
   * Fine-tuning on best samples
   * Shorter training cycle
   * Preservation of reasoning structure
   * Integration of expansions

#### Stage 4: Final RL

Comprehensive reinforcement learning:

1. **Policy Updates**:
   * KL-constrained optimization
   * Reference model comparison
   * Stable policy improvement
   * Expansion-aware updates

2. **Monitoring**:
   * Reward tracking
   * Loss curves
   * Policy divergence checks
   * Expansion utilization metrics

#### Stage 5: Optional Distillation

Knowledge transfer to smaller models:

1. **Student Selection**:
   * Smaller Qwen2.5 variants
   * Architecture preservation
   * Memory optimization
   * Expansion handling capability

2. **Training Process**:
   * Teacher prediction generation
   * Student mimicry learning
   * Checkpoint management
   * CoT structure preservation

### 3. Implementation Details

1. **Memory Efficiency**:
   * Gradient checkpointing by default
   * Automatic mixed precision (AMP)
   * Dynamic batch sizing
   * Efficient attention patterns

2. **Training Stability**:
   * Group advantage normalization
   * KL divergence constraints
   * Reference model comparisons
   * Progress monitoring

3. **Current Status**:
   Our implementation demonstrates:
   * Core DeepSeek-R1 concepts
   * Anthropic expansion integration
   * Starting point for experimentation
   * Learning from paper methodology

### 4. Distillation Implementation

Enhanced knowledge distillation approach:

* Teacher: Trained Qwen2.5-7B model
* Student: Smaller Qwen2.5 variants (1.5B to 32B)
* Training: Supervised learning with CoT preservation
* Focus: Maintaining reasoning capabilities

## Requirements

1. **Python 3.8+**

2. **GPU** (recommended for RL):
   * Minimum: Single GPU with 24GB VRAM
   * Recommended: 40GB+ VRAM (A40, A100)
   * CPU: 32+ cores
   * RAM: 64GB+

3. **API Keys**:
   ```bash
   # DeepSeek API for CoT generation
   export DEEPSEEK_API_KEY="your-key-here"

   # Anthropic API for expansions
   export ANTHROPIC_API_KEY="your-key-here"
   ```

4. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

1. **Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. **API Setup**:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **Run Pipeline**:
   ```bash
   python deepseek_qwen2_5_integration_r1.py
   ```

### Advanced Usage

#### Custom Prompts

Modify prompts in `main()`:

```python
prompts = [
    "Explain quantum entanglement",
    "Solve the traveling salesman problem",
    "Derive the quadratic formula"
]
```

#### API Integration

1. **DeepSeek Usage**:
   ```python
   # Extract both reasoning and final answer
   reasoning_cot = choice.reasoning_content  # Contains <think> tags
   final_text = choice.content  # Final answer only
   ```

2. **Anthropic Integration**:
   ```python
   # Expand uncertain steps
   if is_uncertain_step(reasoning_text):
       expansion = call_anthropic_expansion(
           client,
           model="claude-3-5-sonnet-20241022",
           raw_thought=reasoning_text
       )
   ```

#### Training Configuration

Key parameters to adjust:

```python
# SFT parameters
supervised_fine_tune(
    epochs=5,          # More epochs for better convergence
    batch_size=4,      # Increase for faster training
    lr=5e-6,          # Lower learning rate for stability
    warmup_ratio=0.1   # Longer warmup for complex tasks
)

# RL parameters
rl_training_grpo(
    num_rl_steps=100,  # More steps for better policy
    group_size=8,      # Larger groups for stable updates
    lr=1e-6,          # Conservative learning rate
    clip_ratio=0.15    # Tighter clipping for safety
)
```

## Single-Script Implementation

This repository provides a complete implementation of the DeepSeek-R1 paper in a single Python script, making it easy to understand and modify. Key features:

1. **All-in-One Design**:
   * Complete pipeline in `deepseek_qwen2_5_integration_r1.py`
   * No complex dependencies or distributed setup required
   * Easy to modify and experiment with

2. **Hardware Requirements**:
   * Minimum: Single GPU with 24GB VRAM (e.g., RTX 3090)
   * Recommended: 40GB+ VRAM (e.g., A40, A100)
   * CPU: 32+ cores recommended
   * RAM: 64GB+ recommended

3. **Training Time Estimates**:
   * Cold-Start SFT: ~2-4 hours
   * Initial RL: ~8-12 hours
   * Rejection Sampling: ~2-3 hours
   * Additional SFT: ~4-6 hours
   * Final RL: ~12-24 hours
   * Optional Distillation: ~6-8 hours per model size

4. **Memory Optimization**:
   * Gradient checkpointing enabled by default
   * Automatic mixed precision (AMP) training
   * Efficient attention implementation
   * Dynamic batch sizing based on available VRAM

5. **Customization Points**:
   * Reward functions in `compute_reward()`
   * Model architectures in policy classes
   * Training hyperparameters in each stage
   * Data collection and filtering strategies

> **Resource Note**: For users with limited GPU resources, the script includes flags to run smaller experiments or skip certain stages. The minimal version can run on a 16GB GPU but with reduced performance.

***

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Requirements](#requirements)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Pipeline Stages](#pipeline-stages)
  * [Stage 0: Hybrid CoT Generation](#stage-0-hybrid-cot-generation)
  * [Stage 1: Cold-Start SFT](#stage-1-cold-start-sft)
  * [Stage 2: Reasoning-Oriented RL](#stage-2-reasoning-oriented-rl)
  * [Stage 3: Rejection Sampling & Additional SFT](#stage-3-rejection-sampling--additional-sft)
  * [Stage 4: Final RL](#stage-4-final-rl)
  * [Stage 5: Optional Distillation](#stage-5-optional-distillation)
* [Key Code Snippets](#key-code-snippets)
* [Advanced Topics](#advanced-topics)
* [Citing & Acknowledgments](#citing--acknowledgments)
* [License](#license)

***

## Overview

**R1 + Super CoT: ChainForge** follows the methodology of **DeepSeek-R1** to enhance a Qwen2.5 model's reasoning abilities via reinforcement learning (RL). We:

1. Retrieve high-quality **chain-of-thought** (CoT) from DeepSeek Reasoner's `reasoning_content`
2. Use it for a "cold-start" **supervised fine-tuning** (SFT)
3. Conduct **Reasoning-Oriented RL** to boost correctness and clarity
4. Utilize **rejection sampling** to pick the best RL outputs
5. Perform **additional SFT** on these curated samples
6. Optionally **distill** the final large model into a smaller Qwen2.5 checkpoint

> **Note**: This is a reference pipeline. For production usage, expand datasets, scale RL steps, and incorporate advanced reward modeling.

***

## Features

* **DeepSeek Reasoner Integration**:
  * Automates CoT collection via `reasoning_content`
  * Properly handles <think> tags in chain-of-thought
  * Maintains clean conversation history without reasoning feedback
* **Qwen2.5-7B** Base Model: Hugging Face model with RoPE and large context support
* **Group-based RL**: A GRPO-like approach for stable reinforcement training
* **Rejection Sampling**: Extracts best RL completions for further SFT
* **Distillation**: Compress final RL knowledge into smaller Qwen2.5 variants

***

## Project Structure

```
.
├── deepseek_qwen2_5_integration_r1.py  # Main pipeline implementation
├── requirements.txt                     # Python dependencies
└── README.md                           # Documentation
```

### Key Components

1. **DeepSeek Integration** (`gather_cot_data_from_deepseek`):
   * Automated CoT collection using `reasoning_content`
   * Proper handling of <think> tags
   * Clean conversation history management
   * Error handling and fallbacks

2. **Dataset Classes**:
   * `ChainOfThoughtDataset`: For initial SFT
   * `MockRLReasoningDataset`: For RL training
   * `AdditionalSFTDataset`: For post-RL fine-tuning

3. **RL Components**:
   * `GRPOTorchPolicy`: Policy wrapper
   * `compute_reward`: Reward function
   * `sample_responses`: Response generation

***

## Usage

### Basic Usage

1. **Setup Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   export ANTHROPIC_API_KEY="your-key-here"
   ```

3. **Run Pipeline**:
   ```bash
   python deepseek_qwen2_5_integration_r1.py
   ```

### Advanced Usage

#### Custom Prompts

Modify `deepseek_prompts` in `main()`:

```python
deepseek_prompts = [
    "Explain quantum entanglement",
    "Solve the traveling salesman problem",
    "Derive the quadratic formula"
]
```

#### DeepSeek API Usage

Important notes for using DeepSeek Reasoner:

1. **Handling `reasoning_content`**:
   ```python
   # Extract both reasoning and final answer
   reasoning_cot = choice.reasoning_content  # Contains <think> tags
   final_text = choice.content  # Final answer only

   # Never feed reasoning_content back into conversation
   messages.append({"role": "assistant", "content": final_text})
   ```

2. **Supported Parameters**:
   ```python
   # Only use these parameters
   response = openai.ChatCompletion.create(
       model="deepseek-reasoner",
       messages=messages,
       max_tokens=1024
   )
   ```

3. **Conversation History**:
   * Only append final answers (`content`)
   * Never include `reasoning_content` in history
   * Keep track of turns properly

#### Hyperparameter Tuning

Key parameters to adjust:

```python
# SFT parameters
supervised_fine_tune(
    epochs=5,          # More epochs for better convergence
    batch_size=4,      # Increase for faster training
    lr=5e-6,          # Lower learning rate for stability
    warmup_ratio=0.1   # Longer warmup for complex tasks
)

# RL parameters
rl_training_grpo(
    num_rl_steps=100,  # More steps for better policy
    group_size=8,      # Larger groups for stable updates
    lr=1e-6,          # Conservative learning rate
    clip_ratio=0.15    # Tighter clipping for safety
)
```

***

## Pipeline Stages

### Stage 0: Hybrid CoT Generation

The pipeline begins by gathering high-quality chain-of-thought data from DeepSeek Reasoner and selectively expanding uncertain steps with Anthropic Claude:

1. **Response Format**:
   ```
   Question: {prompt}
   <reasoning_process>
     <think>Step-by-step logical deduction</think>
     <explanation>Anthropic expansion for uncertain steps</explanation>
   </reasoning_process>
   <summary>
     Final concise answer
   </summary>
   ```

2. **API Integration**:
   * DeepSeek Reasoner for base CoT
   * Anthropic Claude for uncertain step expansion
   * Clean conversation history
   * Automatic uncertainty detection

3. **Error Handling**:
   * API failures trigger fallbacks
   * Rate limiting protection
   * Response validation
   * Expansion integration checks

### Stage 1: Cold-Start SFT

Initial supervised fine-tuning on enhanced CoT data:

1. **Data Processing**:
   * Tokenization with proper padding
   * Sequence length management
   * Batch collation
   * Expansion preservation

2. **Training Loop**:
   * Linear learning rate warmup
   * Gradient clipping
   * Progress tracking
   * Validation of reasoning structure

### Stage 2: Reasoning-Oriented RL

Group-based Reward Policy Optimization (GRPO):

1. **Policy Architecture**:
   * Language model as base policy
   * Token-level probability computation
   * Group advantage estimation
   * KL divergence constraints

2. **Reward Structure**:
   * +1.0 for correct answers
   * +0.2 for proper reasoning format
   * Bonus for utilizing expansions
   * Normalized advantages within groups

### Stage 3: Rejection Sampling & Additional SFT

Quality-focused data augmentation:

1. **Sampling Strategy**:
   * Multiple candidates per question
   * Temperature-controlled generation
   * Reward-based filtering
   * Expansion preservation check

2. **Additional Training**:
   * Fine-tuning on best samples
   * Shorter training cycle
   * Preservation of reasoning structure
   * Integration of expansions

### Stage 4: Final RL

Comprehensive reinforcement learning:

1. **Policy Updates**:
   * KL-constrained optimization
   * Reference model comparison
   * Stable policy improvement
   * Expansion-aware updates

2. **Monitoring**:
   * Reward tracking
   * Loss curves
   * Policy divergence checks
   * Expansion utilization metrics

### Stage 5: Optional Distillation

Knowledge transfer to smaller models:

1. **Student Selection**:
   * Smaller Qwen2.5 variants
   * Architecture preservation
   * Memory optimization
   * Expansion handling capability

2. **Training Process**:
   * Teacher prediction generation
   * Student mimicry learning
   * Checkpoint management
   * CoT structure preservation

## Advanced Topics

### Scaling Up

1. **Distributed Training**:
   ```python
   # Add to model configuration
   device_map = "auto"  # or specific device mapping
   ```

2. **Dataset Expansion**:
   * Collect more DeepSeek CoT samples
   * Gather targeted Anthropic expansions
   * Implement custom reward models
   * Add task-specific datasets

### Memory Management

1. **Gradient Checkpointing**:
   ```python
   model.gradient_checkpointing_enable()
   ```

2. **Mixed Precision**:
   ```python
   from torch.cuda.amp import autocast

   with autocast():
       outputs = model(input_ids)
   ```

### Custom Rewards

Implement domain-specific rewards:

```python
def compute_domain_reward(response, ground_truth, has_expansion=False):
    reward = base_reward(response, ground_truth)
    if has_expansion:
        reward *= 1.1  # Bonus for utilizing expansions
    reward += domain_specific_score(response)
    return reward
```

## Citing & Acknowledgments

If you use this code, please cite:

```bibtex
@misc{deepseek2024r1,
  title={DeepSeek-R1: Augmenting Reasoning via Reinforcement Learning},
  author={DeepSeek Team},
  year={2024},
  publisher={arXiv}
}
```

### Contributors

* Nicolas W Schlaepfer (Initial Implementation)
* DeepSeek Team (Original R1 Methodology)
* Qwen Team (Base Models)
* Anthropic (Claude Integration)

## License

MIT License. See [LICENSE](LICENSE) for details.

## 🔥 Local Qwen 3 Agent (Apple Silicon)

> New in `qwen3-agent` branch

1. `brew install cmake rust`
2. `git submodule update --init`
3. `bash scripts/build_llama_metal.sh`
4. `pip install -e .`
5. Download **Qwen3-14B-Q4_K_M.gguf** ➜ `~/models`
6. `bash scripts/run_agent.sh`

The `run_agent.sh` script exports the config path and launches `python -m agent.main`. Users on Intel Macs can unset `LLAMA_METAL` to force CPU inference.
