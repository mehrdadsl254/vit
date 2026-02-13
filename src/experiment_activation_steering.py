"""
Activation Steering Experiment with Qwen2.5-VL-3B-Instruct
===========================================================

This script steers visual activations inside the LLM decoder so that
patches corresponding to a red car are shifted toward the "blue" embedding,
then observes whether the model's answer changes.

Two interpretability approaches:
  1. Logit Lens  – use the model's lm_head (unembedding matrix) to obtain
                   token embeddings for "blue" / "red" and compute the
                   steering direction.
  2. Tuned Lens  – (future) train affine probes per layer.

Usage:
    python src/experiment_activation_steering.py \
        --image_path /path/to/image.png \
        --model_path /home/mmd/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct \
        --base_prompt "What do you see in the picture?" \
        --steer_prompt "What is the color of the car?" \
        --target_word blue \
        --source_word red \
        --layers 14 \
        --alphas 1.0 5.0 10.0 20.0 \
        --output_dir ./results
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Activation Steering with Qwen2.5-VL")
    p.add_argument("--image_path", type=str, required=True,
                    help="Path to input image (e.g. image.png)")
    p.add_argument("--model_path", type=str,
                    default="/home/mmd/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct",
                    help="Local path or HuggingFace ID for Qwen2.5-VL")
    p.add_argument("--base_prompt", type=str,
                    default="What do you see in the picture?",
                    help="Prompt for the base (unsteered) run")
    p.add_argument("--steer_prompt", type=str,
                    default="What is the color of the car?",
                    help="Prompt for the steered run")
    p.add_argument("--target_word", type=str, default="blue",
                    help="Word whose embedding we steer *toward*")
    p.add_argument("--source_word", type=str, default="red",
                    help="Word whose embedding we steer *away from*")
    p.add_argument("--layers", type=int, nargs="+", default=[14],
                    help="LLM decoder layer indices to hook")
    p.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 5.0, 10.0, 20.0],
                    help="Steering strengths to sweep")
    p.add_argument("--output_dir", type=str, default="./results",
                    help="Directory to save results")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(model_path: str):
    """Load the Qwen2.5-VL model and processor."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading model from {model_path} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path)
    print("Model and processor loaded.")
    return model, processor


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_inputs(processor, image_path: str, prompt: str):
    """Build model inputs for a single image + text prompt."""
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def generate_answer(model, processor, inputs, max_new_tokens=128):
    """Run generation and decode the answer."""
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


# ---------------------------------------------------------------------------
# Car-patch segmentation (simple colour threshold for blue-background image)
# ---------------------------------------------------------------------------

def segment_car_patches(image_path: str, patch_size: int = 14):
    """
    Identify which patches belong to the car vs. background.

    Since the background is solid blue (roughly RGB ≈ (60, 60, 200)),
    we threshold: a patch is "car" if its average blue-channel ratio
    is low enough (i.e. it's NOT dominantly blue).

    Returns
    -------
    car_mask : np.ndarray[bool]  – flat array, True for car patches
    grid_h, grid_w : int
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    H, W, _ = img.shape

    # Qwen2.5-VL resizes to multiples of 28 and uses 14-px patches
    # The processor handles the actual resize, but we approximate here
    # to get a rough patch grid for masking.
    grid_h = H // patch_size
    grid_w = W // patch_size

    car_mask = np.zeros((grid_h, grid_w), dtype=bool)

    for i in range(grid_h):
        for j in range(grid_w):
            patch = img[i * patch_size:(i + 1) * patch_size,
                        j * patch_size:(j + 1) * patch_size]
            mean_rgb = patch.mean(axis=(0, 1))  # [R, G, B]
            r, g, b = mean_rgb

            # "Blue background" heuristic:
            #   blue channel dominant AND red/green are low
            is_blue_bg = (b > 150) and (r < 120) and (g < 120)
            car_mask[i, j] = not is_blue_bg

    n_car = car_mask.sum()
    n_total = grid_h * grid_w
    print(f"Segmentation: {n_car}/{n_total} patches classified as car "
          f"(grid {grid_h}×{grid_w})")
    return car_mask.flatten(), grid_h, grid_w


# ---------------------------------------------------------------------------
# Logit-Lens embedding extraction
# ---------------------------------------------------------------------------

def get_token_embedding(model, processor, word: str):
    """
    Get the embedding vector for *word* from the model's lm_head
    (i.e. the unembedding / output projection matrix).

    Logit Lens idea: lm_head.weight[token_id] is the direction in hidden
    space that maximally activates that token's logit.
    """
    token_ids = processor.tokenizer.encode(word, add_special_tokens=False)
    if len(token_ids) == 0:
        raise ValueError(f"Word '{word}' could not be tokenised.")
    # Take the first sub-token if the word is split
    tid = token_ids[0]
    token_str = processor.tokenizer.decode([tid])
    print(f"  Token for '{word}': id={tid}, decoded='{token_str}'")

    # lm_head.weight has shape [vocab_size, hidden_dim]
    emb = model.lm_head.weight[tid].detach().clone().float()
    return emb


# ---------------------------------------------------------------------------
# Activation steering hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """
    A callable that can be registered as a forward hook on a decoder layer.
    It shifts hidden states at *image-patch* positions in the given direction.
    """

    def __init__(self, direction: torch.Tensor, alpha: float,
                 image_token_positions: list):
        """
        Parameters
        ----------
        direction : (hidden_dim,) – unit-norm steering direction
        alpha : float – steering strength
        image_token_positions : list[int] – sequence positions of image tokens
                                             that belong to the car
        """
        self.direction = direction
        self.alpha = alpha
        self.positions = image_token_positions
        self.active = True

    def __call__(self, module, input, output):
        if not self.active:
            return output

        # output can be a tuple; hidden_states is usually the first element
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # hidden_states shape: [batch, seq_len, hidden_dim]
        direction = self.direction.to(hidden_states.device, hidden_states.dtype)
        for pos in self.positions:
            if pos < hidden_states.shape[1]:
                hidden_states[0, pos, :] += self.alpha * direction

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states


# ---------------------------------------------------------------------------
# Find image-token positions in the input sequence
# ---------------------------------------------------------------------------

def find_image_token_positions(inputs, processor, car_mask_flat):
    """
    In Qwen2.5-VL the image is tokenised into a sequence of <|image_pad|>
    tokens inside the full input_ids.  We need to:
      1. Find all positions of <|image_pad|> tokens.
      2. Map car_mask onto those positions.

    Returns list of sequence positions that correspond to car patches.
    """
    input_ids = inputs.input_ids[0]  # [seq_len]

    # Find the image-pad token id
    image_pad_token = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_pad_token is None:
        # fallback: search common token names
        for name in ["<|image_pad|>", "<image>", "<|vision_start|>"]:
            tid = processor.tokenizer.convert_tokens_to_ids(name)
            if tid is not None and tid != processor.tokenizer.unk_token_id:
                image_pad_token = tid
                break

    if image_pad_token is None:
        raise RuntimeError("Could not find image pad token in tokenizer vocab.")

    # All positions of image tokens in the sequence
    image_positions = (input_ids == image_pad_token).nonzero(as_tuple=True)[0].tolist()
    n_image_tokens = len(image_positions)
    print(f"Found {n_image_tokens} image tokens in sequence (total seq len={len(input_ids)})")

    # The car_mask_flat has len = grid_h * grid_w (from our segmentation).
    # The actual number of image tokens may differ because the processor
    # does its own resizing.  We need to map between the two grids.

    n_mask = len(car_mask_flat)
    if n_image_tokens == n_mask:
        # Perfect match
        car_positions = [image_positions[i]
                         for i in range(n_mask) if car_mask_flat[i]]
    elif n_image_tokens > 0:
        # Approximate: linearly map mask indices → image-token indices
        ratio = n_mask / n_image_tokens
        car_positions = []
        for img_idx, seq_pos in enumerate(image_positions):
            mask_idx = int(img_idx * ratio)
            if mask_idx < n_mask and car_mask_flat[mask_idx]:
                car_positions.append(seq_pos)
    else:
        car_positions = []

    print(f"Car-patch positions in sequence: {len(car_positions)} / {n_image_tokens}")
    return car_positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model --------------------------------------------------------
    model, processor = load_model_and_processor(args.model_path)

    # ---- Base run (no intervention) ----------------------------------------
    print("\n" + "=" * 60)
    print("BASE RUN (no steering)")
    print("=" * 60)

    inputs_base = build_inputs(processor, args.image_path, args.base_prompt)
    answer_base = generate_answer(model, processor, inputs_base,
                                  max_new_tokens=args.max_new_tokens)
    print(f"  Prompt : {args.base_prompt}")
    print(f"  Answer : {answer_base}")

    inputs_color = build_inputs(processor, args.image_path, args.steer_prompt)
    answer_color_base = generate_answer(model, processor, inputs_color,
                                        max_new_tokens=args.max_new_tokens)
    print(f"  Prompt : {args.steer_prompt}")
    print(f"  Answer : {answer_color_base}")

    # ---- Segment car patches -----------------------------------------------
    print("\n" + "=" * 60)
    print("SEGMENTATION")
    print("=" * 60)
    car_mask_flat, grid_h, grid_w = segment_car_patches(args.image_path)

    # ---- Logit Lens: get steering direction --------------------------------
    print("\n" + "=" * 60)
    print("LOGIT LENS – Extracting embeddings")
    print("=" * 60)
    emb_target = get_token_embedding(model, processor, args.target_word)
    emb_source = get_token_embedding(model, processor, args.source_word)

    direction = emb_target - emb_source
    direction = direction / direction.norm()  # unit normalise
    print(f"  Steering direction norm (before normalisation): "
          f"{(emb_target - emb_source).norm().item():.4f}")

    # ---- Find image-token positions ----------------------------------------
    # Re-build inputs for the steer prompt (same as color prompt)
    inputs_steer = build_inputs(processor, args.image_path, args.steer_prompt)
    car_positions = find_image_token_positions(inputs_steer, processor,
                                               car_mask_flat)

    if len(car_positions) == 0:
        print("WARNING: No car-patch positions found. Steering will have no effect.")

    # ---- Determine available layers ----------------------------------------
    # Qwen2.5-VL LLM decoder layers are at model.model.layers
    n_layers = len(model.model.layers)
    print(f"\nModel has {n_layers} decoder layers.")

    # ---- Steered runs ------------------------------------------------------
    results = []
    for layer_idx in args.layers:
        if layer_idx >= n_layers:
            print(f"Skipping layer {layer_idx} (only {n_layers} layers).")
            continue

        for alpha in args.alphas:
            print(f"\n{'─' * 60}")
            print(f"STEERED RUN | layer={layer_idx}  alpha={alpha}")
            print(f"{'─' * 60}")

            # Register hook
            hook_obj = SteeringHook(direction, alpha, car_positions)
            target_layer = model.model.layers[layer_idx]
            handle = target_layer.register_forward_hook(hook_obj)

            try:
                # Re-build fresh inputs each time
                inputs_s = build_inputs(processor, args.image_path,
                                        args.steer_prompt)
                answer_steered = generate_answer(
                    model, processor, inputs_s,
                    max_new_tokens=args.max_new_tokens)
            finally:
                handle.remove()

            print(f"  Prompt  : {args.steer_prompt}")
            print(f"  Answer  : {answer_steered}")

            results.append({
                "layer": layer_idx,
                "alpha": alpha,
                "base_answer": answer_color_base,
                "steered_answer": answer_steered,
            })

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Layer':<8} {'Alpha':<10} {'Base Answer':<40} {'Steered Answer'}")
    print("-" * 100)
    for r in results:
        base_short = r["base_answer"][:37] + "..." if len(r["base_answer"]) > 40 else r["base_answer"]
        steer_short = r["steered_answer"][:60]
        print(f"{r['layer']:<8} {r['alpha']:<10.1f} {base_short:<40} {steer_short}")

    # Save to file
    out_file = os.path.join(args.output_dir, "steering_results.json")
    summary = {
        "image_path": args.image_path,
        "base_prompt": args.base_prompt,
        "steer_prompt": args.steer_prompt,
        "target_word": args.target_word,
        "source_word": args.source_word,
        "base_answer_describe": answer_base,
        "base_answer_color": answer_color_base,
        "steered_runs": results,
    }
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
