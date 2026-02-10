from pathlib import Path

from model.listfm_it import load_from_ckpt
from params import config
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader
from datawrapper.datawrapper import LoaderConfig, get_data_wrapper_loader, _coerce_matlab_text
from model.listfm_backbone.module.tokenizer.simple_tokenizer import SimpleTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ========================
# LLM test
# ========================

model_path = "/home/intern4/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
lora_path: str = "/home/intern4/fm2026/mrfm-intern/code_it_llm/dapt/outputs/lora/checkpoint-606"

model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    dtype="auto",
    device_map="auto",
    local_files_only=True
)

model = PeftModel.from_pretrained(model, lora_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path),
    local_files_only=True
)

prompt = "T1 goes to T2 MRI modality, and t2 goes to flair. then does t1 goes to flair?"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Prompt:")
print(prompt)
print("\nResponse:")
print(response)

# ========================
# tokenizer test
# ========================
# tokenizer = SimpleTokenizer()
# file_path = "/fast_storage/intern/data/instruction_tuning/oasis3_longitudinal_mat_new"
# file_list = sorted(Path(file_path).rglob("*.mat"))
# idx = 1
# instruction = loadmat(file_list[idx])["instruction"][0][0]
# print(instruction)
# instruction = _coerce_matlab_text(instruction)
# instruction_token = tokenizer.tokenize(instruction, context_length=64).squeeze()
# print(instruction_token)


# ========================
# data loading test
# ========================
# loader_cfg = LoaderConfig(
#     batch=1,
#     num_workers=2,
#     shuffle=True,
#     debug_mode=True,
#     acs_num=config.acs_num,
#     parallel_factor=config.parallel_factor,
#     data_type=config.data_type,
#     subject_num=config.subject_num,
#     train_percent=config.train_percent,
#     slice_per_subject=config.slice_per_subject,
#     qwen_model_path=config.qwen_model_path,
#     qwen_max_length=config.qwen_max_length,
#     qwen_use_fast=config.qwen_use_fast,
# )

# train_loader, dataset, _ = get_data_wrapper_loader(
#     file_path=["/fast_storage/intern/data/instruction_tuning/oasis3_identity_mat/train"],
#     training_mode=True,
#     loader_cfg=loader_cfg,
#     split="train",
# )

# for batch_idx, _data in enumerate(train_loader):
#     print(f"Batch {batch_idx}:")
#     print(f"min input: {_data[0].min()}", f"max input: {_data[0].max()}")
#     print(f"input dtype: {_data[0].dtype}", f"shape: {_data[0].shape}")
#     print(f"min input: {_data[1].min()}", f"max input: {_data[1].max()}")
#     print(f"label dtype: {_data[1].dtype}", f"shape: {_data[1].shape}")
#     print(f"text_token dtype: {_data[2].dtype}", f"shape: {_data[2].shape}")
#     print(f"instruction_token dtype: {_data[3].dtype}", f"shape: {_data[3].shape}")
#     print(f"instruction_llm_ids dtype: {_data[4].dtype}", f"shape: {_data[4].shape}")
#     print(f"instruction_llm_mask dtype: {_data[5].dtype}", f"shape: {_data[5].shape}")
#     break


# ========================
# model loading test
# ========================
# model = load_from_ckpt(
#     ckpt_path=Path(config.pretrained),
#     from_scratch=config.from_scratch,
#     use_vision_decoder=True,
#     use_vision_decoder_weights=False,
# )

# listfmconfig = model.listfmconfig

# vision_encoder_params = sum(p.numel() for p in model.vision_encoder.parameters())
# vision_decoder_params = sum(p.numel() for p in model.vision_decoder.parameters())
# print(f"Vision Encoder Parameters: {vision_encoder_params}")
# print(f"Vision Decoder Parameters: {vision_decoder_params}")

# print(listfmconfig)
