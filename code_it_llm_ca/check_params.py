import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.listfm_it import load_from_ckpt
from params import config as train_config


def format_number(num):
    """Format large numbers with commas and suffix"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))


def print_params(model, ckpt_path=None):
    """
    Print model parameter statistics
    
    Args:
        model: PyTorch model
        ckpt_path: Path to checkpoint (for display)
        show_breakdown: Whether to show module breakdown
    """
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL PARAMETERS SUMMARY")
    print("="*80)
    if ckpt_path:
        print(f"Checkpoint: {ckpt_path}")
    print("-"*80)
    print(f"Total Parameters:         {total_params:>18,} ({format_number(total_params):>8})")
    print(f"  ├─ Trainable:           {trainable_params:>18,} ({format_number(trainable_params):>8}) {trainable_params/total_params*100:>6.2f}% ✓")
    print(f"  └─ Frozen:              {frozen_params:>18,} ({format_number(frozen_params):>8}) {frozen_params/total_params*100:>6.2f}% ✗")
    print("="*80)
    
    # LoRA breakdown (by parameter name)
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n.lower()]
    if lora_params:
        lora_total = sum(p.numel() for _, p in lora_params)
        lora_trainable = sum(p.numel() for _, p in lora_params if p.requires_grad)
        lora_frozen = lora_total - lora_trainable
        print("\nLORA PARAMETERS SUMMARY")
        print("-"*80)
        print(f"LoRA Parameters:          {lora_total:>18,} ({format_number(lora_total):>8})")
        print(f"  ├─ Trainable:           {lora_trainable:>18,} ({format_number(lora_trainable):>8}) {lora_trainable/lora_total*100:>6.2f}% ✓")
        print(f"  └─ Frozen:              {lora_frozen:>18,} ({format_number(lora_frozen):>8}) {lora_frozen/lora_total*100:>6.2f}% ✗")
        print("-"*80)
    else:
        print("\nLORA PARAMETERS SUMMARY")
        print("-"*80)
        print("No LoRA parameters found by name filter ('lora').")
        print("-"*80)

    # Module breakdown
    print("\nBREAKDOWN BY MODULE:")
    print("-"*80)
    
    module_params = {}
    for name, param in model.named_parameters():
        # Get top-level module name
        module_name = name.split('.')[0]
        if module_name not in module_params:
            module_params[module_name] = {
                'total': 0,
                'trainable': 0,
                'frozen': 0,
                'details': []
            }
        
        param_count = param.numel()
        is_trainable = param.requires_grad
        
        module_params[module_name]['total'] += param_count
        if is_trainable:
            module_params[module_name]['trainable'] += param_count
        else:
            module_params[module_name]['frozen'] += param_count
        
        module_params[module_name]['details'].append({
            'name': name,
            'count': param_count,
            'trainable': is_trainable
        })
    
    # Print module summary
    for module_name in sorted(module_params.keys()):
        stats = module_params[module_name]
        total = stats['total']
        trainable = stats['trainable']
        frozen = stats['frozen']
        ratio = trainable / total * 100 if total > 0 else 0
        
        status = "✓" if frozen == 0 else ("✗" if trainable == 0 else "◐")
        print(f"\n  {module_name:<35} {total:>15,} ({format_number(total):>8}) {status}")
        print(f"    ├─ Trainable:   {trainable:>15,} ({format_number(trainable):>8}) {ratio:>6.2f}%")
        if frozen > 0:
            print(f"    └─ Frozen:      {frozen:>15,} ({format_number(frozen):>8}) {100-ratio:>6.2f}%")
    
    print("-"*80)

    print()


if __name__ == "__main__":
    config_ckpt = Path(train_config.pretrained)
    if not config_ckpt.exists():
        print(f"Error: Config checkpoint not found: {config_ckpt}")
        sys.exit(1)

    print("Initializing model exactly like train.py (no weights load).")
    print(f"Model config checkpoint: {config_ckpt}")
    if train_config.qwen_model_path:
        print(f"Using Qwen model: {train_config.qwen_model_path}")
        if train_config.qwen_lora_path:
            print(f"Using Qwen LoRA: {train_config.qwen_lora_path}")

    model = load_from_ckpt(
        ckpt_path=config_ckpt,
        from_scratch=train_config.from_scratch,
        use_vision_decoder=True,
        use_vision_decoder_weights=False,
        qwen_model_path=train_config.qwen_model_path if train_config.qwen_model_path else None,
        qwen_lora_path=train_config.qwen_lora_path if train_config.qwen_lora_path else None,
        qwen_trainable=train_config.qwen_trainable,
    )
    
    print_params(model, ckpt_path=config_ckpt)
