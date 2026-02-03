from pathlib import Path

from model.listfm_it import load_from_ckpt
from params import config

    
model = load_from_ckpt(
    ckpt_path=Path(config.pretrained),
    from_scratch=config.from_scratch,
    use_vision_decoder=False,  # We only need text encoder
    use_vision_decoder_weights=False,
)

listfmconfig = model.listfmconfig

print(listfmconfig)