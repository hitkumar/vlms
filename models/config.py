from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = (
        "google/siglip-base-patch16-512"  # 'google/siglip-base-patch16-224'
    )

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49280
    extra_token_amount: int = (
        1  # number of extra tokens for the vlm (image_start, image_end, etc.)
    )
    lm_vocab_size: int = lm_base_vocab_size + extra_token_amount
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 1024
    lm_use_tokens: bool = (
        False  # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    )
    lm_tie_weights: bool = (
        False  # Decide if you want to tie the LM Head weight to the token embeding weights
    )
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"
    hf_repo_name: str = "nanoVLM"
    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {"image_token": "<|image|>"}
    )
    lm_chat_template: str = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )


@dataclass
class TrainConfig:
    lr_mp: float = 1e-4
    lr_backbones: float = 5e-5
    data_cutoff_idx: int = None
    batch_size: int = 32
    mmstar_batch_size: int = 32
    eval_in_epochs: bool = True
    epochs: int = 1
    compile: bool = True
    resume_from_vlm_checkpoint: bool = (
        False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    )
    train_dataset_path: str = "HuggingFaceM4/the_cauldron"
    train_dataset_name: tuple[str, ...] = (
        "ai2d",
        "aokvqa",
        "chart2text",
        "chartqa",
        "clevr",
        "cocoqa",
        "datikz",
        "diagram_image_to_text",
        "docvqa",
        "dvqa",
        "figureqa",
        "finqa",
        "geomverse",
        "hateful_memes",
        "hitab",
        "iam",
        "iconqa",
        "infographic_vqa",
        "intergps",
        "localized_narratives",
        "mapqa",
        "multihiertt",
        "ocrvqa",
        "plotqa",
        "raven",
        "rendered_text",
        "robut_sqa",
        "robut_wikisql",
        "robut_wtq",
        "scienceqa",
        "screen2words",
        "st_vqa",
        "tabmwp",
        "tallyqa",
        "tat_qa",
        "textcaps",
        "textvqa",
        "tqa",
        "vistext",
        "visual7w",
        "visualmrc",
        "vqarad",
        "vqav2",
        "vsr",
        "websight",
    )  # "clevr_math", "okvqa", "spot_the_diff", "nlvr2", "mimic_cgd",
    test_dataset_path: str = "Lin-Chen/MMStar"
    log_wandb: bool = True
