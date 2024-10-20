import os
from copy import deepcopy

from omegaconf.dictconfig import DictConfig

from .configs import CONFIG_2_0, CONFIG_2_1
from .kandinsky2_1_model import Kandinsky2_1
from .kandinsky2_2_model import Kandinsky2_2
from .kandinsky2_model import Kandinsky2

USE_NEW_API = False
if USE_NEW_API:
    from huggingface_hub import hf_hub_download

    LOCAL_DIR = "./local/kandinsky2"
else:
    from huggingface_hub import cached_download, hf_hub_download, hf_hub_url


def get_kandinsky2_0(
    device,
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
):
    cache_dir = os.path.join(cache_dir, "2_0")
    config = deepcopy(CONFIG_2_0)
    if task_type == "inpainting":
        model_name = "Kandinsky-2-0-inpainting.pt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.0", filename=model_name
        )
    elif task_type == "text2img":
        model_name = "Kandinsky-2-0.pt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.0", filename=model_name
        )
    else:
        raise ValueError("Only text2img, img2img and inpainting is available")

    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=use_auth_token,
    )

    cache_dir_text_en1 = os.path.join(cache_dir, "text_encoder1")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.0", filename=f"text_encoder1/{name}"
        )
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en1,
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    cache_dir_text_en2 = os.path.join(cache_dir, "text_encoder2")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "spiece.model",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.0", filename=f"text_encoder2/{name}"
        )
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en2,
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.0", filename="vae.ckpt"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="vae.ckpt",
        use_auth_token=use_auth_token,
    )

    config["text_enc_params1"]["model_path"] = cache_dir_text_en1
    config["text_enc_params2"]["model_path"] = cache_dir_text_en2
    config["tokenizer_name1"] = cache_dir_text_en1
    config["tokenizer_name2"] = cache_dir_text_en2
    config["image_enc_params"]["params"]["ckpt_path"] = os.path.join(
        cache_dir, "vae.ckpt"
    )
    unet_path = os.path.join(cache_dir, model_name)

    model = Kandinsky2(config, unet_path, device, task_type)
    return model


def get_kandinsky2_1_new(
    task_type="text2img",
    local_dir="/tmp/kandinsky2",
    use_auth_token=None,
    use_flash_attention=False,
    device="cpu",
):
    local_dir = os.path.join(local_dir, "2_1")
    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    repo_id = "sberbank-ai/Kandinsky_2.1"
    if task_type == "text2img":
        model_name = "decoder_fp16.ckpt"
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
    hf_hub_download(
        repo_id=repo_id,
        filename=model_name,
        force_filename=model_name,
        use_auth_token=use_auth_token,
        local_dir=local_dir,
    )
    prior_name = "prior_fp16.ckpt"
    repo_id = "sberbank-ai/Kandinsky_2.1"

    hf_hub_download(
        repo_id=repo_id,
        filename=prior_name,
        force_filename=prior_name,
        use_auth_token=use_auth_token,
        local_dir=local_dir,
    )

    local_dir_text_en = os.path.join(local_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        repo_id = "sberbank-ai/Kandinsky_2.1"
        filename = f"text_encoder/{name}"
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            use_auth_token=use_auth_token,
            local_dir=local_dir,
        )

    repo_id = "sberbank-ai/Kandinsky_2.1"
    filename = "movq_final.ckpt"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        use_auth_token=use_auth_token,
        local_dir=local_dir,
    )

    repo_id = "sberbank-ai/Kandinsky_2.1"
    filename = "ViT-L-14_stats.th"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        use_auth_token=use_auth_token,
        local_dir=local_dir,
    )

    config["tokenizer_name"] = local_dir_text_en
    config["text_enc_params"]["model_path"] = local_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(local_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(local_dir, "movq_final.ckpt")
    local_model_name = os.path.join(local_dir, model_name)
    local_prior_name = os.path.join(local_dir, prior_name)

    model = Kandinsky2_1(
        config, local_model_name, local_prior_name, device, task_type=task_type
    )

    return model


def get_kandinsky2_1(
    device,
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
    use_flash_attention=False,
):
    cache_dir = os.path.join(cache_dir, "2_1")
    config = DictConfig(deepcopy(CONFIG_2_1))
    config["model_config"]["use_flash_attention"] = use_flash_attention
    if task_type == "text2img":
        model_name = "decoder_fp16.ckpt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
        )
    elif task_type == "inpainting":
        model_name = "inpainting_fp16.ckpt"
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=model_name
        )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=model_name,
        use_auth_token=use_auth_token,
    )
    prior_name = "prior_fp16.ckpt"
    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename=prior_name
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename=prior_name,
        use_auth_token=use_auth_token,
    )

    cache_dir_text_en = os.path.join(cache_dir, "text_encoder")
    for name in [
        "config.json",
        "pytorch_model.bin",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        config_file_url = hf_hub_url(
            repo_id="sberbank-ai/Kandinsky_2.1", filename=f"text_encoder/{name}"
        )
        cached_download(
            config_file_url,
            cache_dir=cache_dir_text_en,
            force_filename=name,
            use_auth_token=use_auth_token,
        )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="movq_final.ckpt"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="movq_final.ckpt",
        use_auth_token=use_auth_token,
    )

    config_file_url = hf_hub_url(
        repo_id="sberbank-ai/Kandinsky_2.1", filename="ViT-L-14_stats.th"
    )
    cached_download(
        config_file_url,
        cache_dir=cache_dir,
        force_filename="ViT-L-14_stats.th",
        use_auth_token=use_auth_token,
    )

    config["tokenizer_name"] = cache_dir_text_en
    config["text_enc_params"]["model_path"] = cache_dir_text_en
    config["prior"]["clip_mean_std_path"] = os.path.join(cache_dir, "ViT-L-14_stats.th")
    config["image_enc_params"]["ckpt_path"] = os.path.join(cache_dir, "movq_final.ckpt")
    cache_model_name = os.path.join(cache_dir, model_name)
    cache_prior_name = os.path.join(cache_dir, prior_name)
    model = Kandinsky2_1(
        config, cache_model_name, cache_prior_name, device, task_type=task_type
    )
    return model


def get_kandinsky2(
    device,
    task_type="text2img",
    cache_dir="/tmp/kandinsky2",
    use_auth_token=None,
    model_version="2.1",
    use_flash_attention=False,
):
    if model_version == "2.0":
        model = get_kandinsky2_0(
            device,
            task_type=task_type,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
        )
    elif model_version == "2.1":
        model = get_kandinsky2_1(
            device,
            task_type=task_type,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            use_flash_attention=use_flash_attention,
        )
    elif model_version == "2.2":
        model = Kandinsky2_2(device=device, task_type=task_type)
    else:
        raise ValueError("Only 2.0 and 2.1 is available")

    return model
