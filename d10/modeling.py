"""Model loading (QLoRA 4-bit), LoRA / freezing setups for E1-E3, and adapter + projector I/O."""
import re
from pathlib import Path

import torch
import yaml

LLM_TARGETS = r"(?:.*\.)?language_model\.layers\.\d+\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGETS = r"(?:.*\.)?visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
PROJECTOR_FILE = "projector.pt"


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def autocast():
    """fp16 autocast on GPU. Also needed at inference: the trained projector may be float32."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available())


def load_processor(model_id):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"      # batched generation needs left padding
    return processor


def load_base(model_id):
    """4-bit NF4 base model. The ViT (with the projector) and `lm_head` are skipped, i.e. they
    stay in fp16: a 4-bit ViT would leave the merger with quantised weights that cannot be
    fully trained.

    transformers matches a skip pattern against the START of the full module name (or as its
    suffix), and the ViT is called `model.visual`. The bare pattern "visual" therefore skips
    nothing and the whole ViT is silently quantised; "model.visual" is the one that works.
    """
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["model.visual", "visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    quantised = [n for n, p in model.named_parameters()
                 if ("visual." in n or n.startswith("lm_head")) and not p.is_floating_point()]
    if quantised:
        raise RuntimeError(f"{len(quantised)} ViT/lm_head tensors were quantised, e.g. {quantised[0]}")
    return model


def ignore_incompatible_torchao():
    """Kaggle ships torchao 0.10, which this peft rejects with an ImportError as soon as LoRA is
    put on a plain nn.Linear (the fp16 ViT in E3). Nothing here uses torchao-quantised layers,
    so peft's torchao dispatcher is told that the package is simply not available."""
    import peft.import_utils
    import peft.tuners.lora.torchao as lora_torchao
    try:
        peft.import_utils.is_torchao_available()
    except ImportError as e:
        print(f"note: {e}; ignoring torchao")
        lora_torchao.is_torchao_available = lambda: False


def is_projector(name):
    return any(k in name for k in PROJECTOR_KEYS)


def lora_targets(model, targets):
    """Regex for LoraConfig.target_modules, plus the module names it matches (for checking)."""
    regex = LLM_TARGETS if targets == "llm" else f"(?:{LLM_TARGETS})|(?:{VIT_TARGETS})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    return regex, matched


def apply_setup(model, cfg):
    """Freeze everything, add LoRA on the LLM (E1, E2) or LLM + ViT (E3), and for E2/E3 also
    unfreeze the full projector in float32. Returns (peft_model, info)."""
    from peft import LoraConfig, get_peft_model
    ignore_incompatible_torchao()
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()            # gradient checkpointing needs inputs with grad

    lora = cfg["lora"]
    regex, matched = lora_targets(model, lora["targets"])
    text_cfg, vis_cfg = model.config.text_config, model.config.vision_config
    expected = text_cfg.num_hidden_layers * 7 + (vis_cfg.depth * 4 if "vit" in lora["targets"] else 0)
    if len(matched) != expected:                  # 252 for E1/E2, 348 for E3 on the real model
        raise RuntimeError(f"LoRA regex matched {len(matched)} modules, expected {expected}")

    peft_model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], target_modules=regex))

    n_proj = 0
    if cfg.get("train_projector"):
        for name, p in peft_model.named_parameters():
            if is_projector(name) and "lora_" not in name:
                p.requires_grad = True
                n_proj += p.numel()
    for p in peft_model.parameters():
        if p.requires_grad:
            # Everything that is optimised lives in float32 (the frozen rest stays fp16/4-bit):
            # GradScaler refuses fp16 gradients, and an update of size lr*grad is below fp16
            # resolution for most weights, so it would be rounded away.
            p.data = p.data.float()
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    info = {"lora_modules": len(matched), "trainable_params": trainable,
            "projector_params": n_proj, "lora_params": trainable - n_proj}
    return peft_model, info


def save_trained(peft_model, out_dir):
    """peft's save_pretrained stores ONLY the LoRA adapter. The fully trained projector is not
    part of it, so it is saved separately; forgetting this makes E2/E3 look exactly like E1."""
    out_dir = Path(out_dir)
    peft_model.save_pretrained(out_dir)
    state = {n: p.detach().cpu() for n, p in peft_model.get_base_model().named_parameters()
             if is_projector(n) and p.requires_grad}
    if state:
        torch.save(state, out_dir / PROJECTOR_FILE)
    return len(state)


def load_projector(base, run_dir):
    path = Path(run_dir) / PROJECTOR_FILE
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(base.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():        # KeyError here = file does not fit this model
            params[name].copy_(tensor.to(params[name].dtype))
    return len(state)


def load_finetuned(model_id, run_dir, trainable=False):
    """Base 4-bit model -> restore trained projector (if any) -> attach the LoRA adapter."""
    from peft import PeftModel
    ignore_incompatible_torchao()
    base = load_base(model_id)
    n_proj = load_projector(base, run_dir)
    if trainable:
        base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, str(run_dir), is_trainable=trainable)
    print(f"loaded adapter from {run_dir} (+{n_proj} projector tensors)")
    return model
