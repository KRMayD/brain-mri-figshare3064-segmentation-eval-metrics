import argparse
import os
import re

import torch

STATE_DICT_PATTERNS = [
    (r"visual\.head.proj.(\w+)", "visual_projection.{0}"),
    (r"visual\.trunk\.norm\.(\w+)", "vision_model.post_layernorm.{0}"),
    (r"visual\.trunk.patch_embed\.proj\.(\w+)", "vision_model.embeddings.patch_embedding.{0}"),
    (r"visual\.trunk\.blocks\.(\w+)\.norm2\.(\w+)", "vision_model.encoder.layers.{0}.layer_norm2.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.norm1\.(\w+)", "vision_model.encoder.layers.{0}.layer_norm1.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.attn\.proj\.(\w+)", "vision_model.encoder.layers.{0}.self_attn.out_proj.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc1\.(\w+)", "vision_model.encoder.layers.{0}.mlp.fc1.{1}"),
    (r"visual\.trunk\.blocks\.(\w+)\.mlp\.fc2\.(\w+)", "vision_model.encoder.layers.{0}.mlp.fc2.{1}"),
    (r"text\.transformer\.embeddings\.token_type_embeddings.(\w+)", "text_model.embeddings.token_type_embedding.{0}"),
    (r"text\.transformer\.embeddings\.word_embeddings.(\w+)", "text_model.embeddings.token_embedding.{0}"),
    (r"text\.transformer\.embeddings\.position_embeddings.(\w+)", "text_model.embeddings.position_embedding.{0}"),
    (r"text\.transformer\.embeddings\.LayerNorm\.(\w+)", "text_model.embeddings.layer_norm.{0}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.key.(\w+)", "text_model.encoder.layers.{0}.self_attn.k_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.query.(\w+)", "text_model.encoder.layers.{0}.self_attn.q_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.self.value.(\w+)", "text_model.encoder.layers.{0}.self_attn.v_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.dense.(\w+)", "text_model.encoder.layers.{0}.self_attn.out_proj.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).attention.output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).intermediate.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc1.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.LayerNorm.(\w+)", "text_model.encoder.layers.{0}.layer_norm2.{1}"),
    (r"text\.transformer\.encoder\.layer\.(\w+).output.dense.(\w+)", "text_model.encoder.layers.{0}.mlp.fc2.{1}"),
    (r"text\.proj\.0\.(\w+)", "text_projection.fc1.{0}"),
    (r"text\.proj\.2\.(\w+)", "text_projection.fc2.{0}"),
]


def convert_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        found = False
        match = re.match(r"visual\.trunk\.blocks\.(\w+)\.attn\.qkv\.(\w+)", key)
        if match:
            chunks = value.chunk(3, dim=0)
            for proj_name, proj_value in zip(["q_proj", "k_proj", "v_proj"], chunks):
                new_key = f"vision_model.encoder.layers.{match.group(1)}.self_attn.{proj_name}.{match.group(2)}"
                new_state_dict[new_key] = proj_value
            continue

        if key == "visual.trunk.cls_token":
            new_state_dict["vision_model.embeddings.class_embedding"] = value.squeeze(0).squeeze(0)
            continue
        if key == "visual.trunk.pos_embed":
            new_state_dict["vision_model.embeddings.position_embedding.weight"] = value.squeeze(0)
            continue

        for pattern, replacement in STATE_DICT_PATTERNS:
            match = re.match(pattern, key)
            if match:
                new_state_dict[replacement.format(*match.groups())] = value
                found = True
                break
        if not found:
            new_state_dict[key] = value
    return new_state_dict


def clean_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[6:]
        if key.startswith("module."):
            key = key[7:]
        new_state_dict[key] = value
    return new_state_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--model-name", default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    openclip_src = os.path.join(base_dir, "biomedclip_finetuning", "open_clip", "src")
    import sys
    if openclip_src not in sys.path:
        sys.path.insert(0, openclip_src)

    from open_clip import create_model_from_pretrained

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = clean_state_dict(state_dict)

    openclip_model, _ = create_model_from_pretrained(args.model_name)
    openclip_model.load_state_dict(state_dict, strict=False)
    hf_state_dict = convert_state_dict(openclip_model.state_dict())

    torch.save(hf_state_dict, args.output_path)
    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()
