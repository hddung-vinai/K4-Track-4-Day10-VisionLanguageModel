"""Training examples: chat template, loss mask, and batching."""
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from d10.labels import PROMPT, objects_to_answer

IGNORE = -100
CHAT_KW = dict(tokenize=True, return_dict=True, return_tensors="pt")


def user_message(image, prompt=PROMPT):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, answer, prompt=PROMPT):
    """Tokenise (image, prompt, answer) and build labels that are -100 everywhere except the
    answer, so the loss is computed only on what the model has to generate."""
    user = user_message(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **CHAT_KW)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **CHAT_KW)
    n = prompt_enc["input_ids"].shape[1]
    # The mask is only correct if the prompt is an exact prefix of the full sequence.
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt tokens are not a prefix of the full sequence")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = IGNORE
    out = dict(full_enc)
    out["labels"] = labels
    return out


class GroundingDataset(torch.utils.data.Dataset):
    def __init__(self, rows, data_dir, processor):
        self.rows, self.data_dir, self.processor = rows, Path(data_dir), processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = Image.open(self.data_dir / row["image_path"]).convert("RGB")
        return encode_example(self.processor, image, objects_to_answer(row["objects"]))


def collate(batch, pad_token_id):
    """Right-pad the per-token tensors to one length (labels with -100). Vision tensors hold
    patches of all images back to back, so they are concatenated, never padded."""
    length = max(b["input_ids"].shape[1] for b in batch)

    def pad(key, value):
        return torch.cat([F.pad(b[key], (0, length - b[key].shape[1]), value=value) for b in batch])

    out = {"input_ids": pad("input_ids", pad_token_id), "attention_mask": pad("attention_mask", 0),
           "labels": pad("labels", IGNORE)}
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch])
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch])
    return out
