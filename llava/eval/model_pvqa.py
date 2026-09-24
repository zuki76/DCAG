import argparse
import json
import os

import numpy as np
import shortuuid
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    KeywordsStoppingCriteria,
)
from llava.eval.coin_utils import get_model_name_from_path
from llava.eval._ad_split import apply_ad_split, should_split_ad_mosaic
from llava.eval.domain_router import resolve_domain_id


DOMAIN_TO_ID = {"RS": 0, "Med": 1, "AD": 2, "Sci": 3, "Fin": 4}


def split_list(lst, n):
    if n <= 0:
        raise ValueError(f"num_chunks must be positive, got {n}")
    return [lst[i * len(lst) // n : (i + 1) * len(lst) // n] for i in range(n)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def pil_to_unit_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0


def maybe_split_ad_mosaic(
    image_tensor: torch.Tensor,
    image_path: str,
    image_folder: str = "",
    domain_name: str = "",
) -> torch.Tensor:
    """Split a driving mosaic with the shared preprocessing rule."""
    if not should_split_ad_mosaic(image_path, image_folder, domain_name):
        return image_tensor
    return apply_ad_split(image_tensor)


def make_domain_image(
    image: Image.Image,
    image_file: str,
    domain_name: str | None,
    image_folder: str = "",
) -> torch.Tensor:
    tensor = pil_to_unit_tensor(image)
    if domain_name == "AD":
        return maybe_split_ad_mosaic(tensor, image_file, image_folder, domain_name)
    return tensor


class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
        domain_name=None,
    ):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.domain_name = domain_name

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"] + " Answer the question with a single word (or phrase)."
        if self.model_config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[
            0
        ]
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        if self.domain_name is None:
            return input_ids, image_tensor

        domain_id = torch.tensor(
            resolve_domain_id(self.domain_name, image), dtype=torch.long
        )
        routed_domain = list(DOMAIN_TO_ID)[int(domain_id)]
        domain_image = make_domain_image(
            image, image_file, routed_domain, self.image_folder
        )
        return input_ids, image_tensor, domain_id, domain_image

    def __len__(self):
        return len(self.questions)


def _pad_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    return 0


def _make_collate(tokenizer, has_domain):
    pad = _pad_id(tokenizer)

    def _collate(batch):
        if has_domain:
            input_ids_list = [b[0] for b in batch]
            image_tensors = [b[1] for b in batch]
            domain_ids = torch.stack([b[2] for b in batch], dim=0)
            domain_images = [b[3] for b in batch]
        else:
            input_ids_list = [b[0] for b in batch]
            image_tensors = [b[1] for b in batch]
            domain_ids = None
            domain_images = None

        try:
            image_tensor = torch.stack(image_tensors, dim=0)
        except Exception:
            image_tensor = image_tensors

        max_len = max(x.shape[0] for x in input_ids_list)
        padded = []
        attn = []
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            padded.append(
                torch.cat([torch.full((pad_len,), pad, dtype=ids.dtype), ids], dim=0)
            )
            attn.append(
                torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(ids.shape[0], dtype=torch.long),
                    ],
                    dim=0,
                )
            )
        input_ids = torch.stack(padded, dim=0)
        attention_mask = torch.stack(attn, dim=0)

        if domain_images is not None:
            try:
                first_shape = domain_images[0].shape
                if all(di.shape == first_shape for di in domain_images):
                    domain_images = torch.stack(domain_images, dim=0)
            except Exception:
                pass

        return input_ids, attention_mask, image_tensor, domain_ids, domain_images

    return _collate


def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    conv_mode,
    batch_size=1,
    num_workers=4,
    domain_name=None,
):
    dataset = CustomDataset(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        conv_mode,
        domain_name=domain_name,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_make_collate(tokenizer, has_domain=(domain_name is not None)),
    )


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print(f"Loading model from {model_path}")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    tokenizer.padding_side = "left"
    model.config.tokenizer_padding_side = "left"

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        domain_name=args.domain_name,
    )

    conv_template = conv_templates[args.conv_mode]
    stop_str = (
        conv_template.sep
        if conv_template.sep_style != SeparatorStyle.TWO
        else conv_template.sep2
    )

    for batch_idx, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, len(questions))
        batch_lines = questions[start:end]

        input_ids, attention_mask, image_tensor, domain_ids, domain_images = batch
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)

        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        generate_kwargs = {
            "attention_mask": attention_mask,
            "do_sample": args.temperature > 0,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
            "max_new_tokens": args.max_new_tokens,
            "stopping_criteria": [stopping_criteria],
            "use_cache": True,
        }
        if torch.is_tensor(image_tensor):
            generate_kwargs["images"] = image_tensor.to(
                dtype=torch.float16, device="cuda", non_blocking=True
            )
        else:
            generate_kwargs["images"] = image_tensor
        if domain_ids is not None:
            generate_kwargs["domain_ids"] = domain_ids.to(
                device="cuda", non_blocking=True
            )
        if domain_images is not None:
            if torch.is_tensor(domain_images):
                generate_kwargs["domain_images"] = domain_images.to(
                    dtype=torch.float16, device="cuda", non_blocking=True
                )
            else:
                generate_kwargs["domain_images"] = [
                    di.to(dtype=torch.float16, device="cuda", non_blocking=True)
                    for di in domain_images
                ]

        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generate_kwargs)

        input_token_len = input_ids.shape[1]
        outputs_batch = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )

        for i, output in enumerate(outputs_batch):
            text = output.split(stop_str)[0].strip()
            line = batch_lines[i]
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": line["question_id"],
                        "prompt": line["text"],
                        "text": text,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--domain-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    eval_model(args)
