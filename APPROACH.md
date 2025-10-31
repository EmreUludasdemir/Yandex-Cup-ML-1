# Hallucination Resistance Approach

## Strategy

This solution implements multiple layers of defense against LLM hallucinations:

### 1. Model Selection
- **Base Model**: Saiga Mistral 7B LoRA - A Russian-optimized instruction-following model
- **Quantization**: 4-bit quantization (NF4) for efficient inference on Tesla L4
- **Benefits**: Good Russian language understanding, instruction following, and fits in GPU memory

### 2. Anti-Hallucination Techniques

#### A. Pattern-Based Detection
- Pre-filters questions for known impossible scenarios:
  - Temporal anachronisms (ancient tech + modern inventions)
  - Physically impossible scenarios
  - Fictional entities treated as real
  - Questions with false premises

#### B. Prompt Engineering
- System prompt emphasizes:
  - Honesty over helpfulness
  - "I don't know" is acceptable and encouraged
  - Explicit examples of impossible questions
  - Short, factual answers only (no elaboration)

#### C. Low-Temperature Sampling
- Temperature: 0.3 (low variance for consistency)
- Top-p: 0.9 (nucleus sampling)
- Top-k: 50 (limited vocabulary at each step)

#### D. Confidence Thresholding
- Calculates confidence from token probabilities
- If confidence < 0.4, returns "Я не знаю"
- Ensures model only answers when certain

#### E. Answer Validation
- Checks for uncertainty markers in generated text
- Detects hedging language (multiple "maybe", "probably", etc.)
- Validates answer length and format
- Filters meta-commentary or refusals

### 3. Consistency Strategy

For the three rephrasings:
- Low temperature ensures similar answers for similar questions
- Conservative confidence threshold means uncertain answers become "Я не знаю" for all three
- This leverages the "honesty bonus" (0.15 points) when truly uncertain

### 4. Scoring Optimization

Target scoring strategy:
- **High confidence factual questions**: Answer correctly (1.0 point)
- **Low confidence factual questions**: Say "Я не знаю" to all 3 rephrasings (0.15 points)
- **Hallucination provocations**: Refuse or say "Я не знаю" (1.0 point)

Expected score breakdown:
- If we can correctly answer 70% of factual questions: 0.7 points
- Honesty bonus on remaining 30%: 0.3 * 0.15 = 0.045 points
- Hallucination resistance at 95%: 0.95 points
- Final: 1000 * (0.8 * 0.745 + 0.2 * 0.95) = 1000 * 0.786 = 786 points

## Technical Constraints

- **GPU**: NVIDIA Tesla L4 (24GB VRAM)
- **Time limit**: 1 hour for ~1200 queries (~3 seconds per query)
- **Docker image**: < 15 GB
- **Memory**: 4-bit quantization uses ~4-5 GB for 7B model

## Performance Characteristics

- **Inference speed**: ~2-3 seconds per query with quantization
- **Memory usage**: ~5-6 GB GPU memory
- **Docker image size**: ~10-12 GB (PyTorch + transformers + model weights)
- **Expected throughput**: ~1200-1800 queries per hour

## Potential Improvements

1. **Fine-tuning**: Fine-tune on examples of "I don't know" for Russian factual Q&A
2. **Ensemble**: Use multiple smaller models and aggregate confidence
3. **Retrieval**: Add RAG (Retrieval-Augmented Generation) for factual grounding
4. **Calibration**: Tune confidence threshold based on validation set
5. **Better detection**: Train a separate classifier for hallucination provocations
