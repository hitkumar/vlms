import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_str = "dummy_content"
        random_str_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_str}],
            tokenize=False,
            add_special_tokens=False,
        )
        random_str_location = random_str_chat_templated.find(random_str)
        return len(
            self.tokenizer.encode(random_str_chat_templated[:random_str_location])
        )

    def _get_messages(self, item, image_count=0):
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        if image_count > 0:
            # Prepend the image tokens to the first message only, this serves as the context for the whole conversation. `mp_image_token_length`` is the number of tokens per image in the MP model.
            messages[0]["content"] = (
                self.tokenizer.image_token * image_count * self.mp_image_token_length
                + messages[0]["content"]
            )

        return messages

    def _process_images(self, images):
        # convert images to tensors
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image = self.image_processor(image)
                processed_images.append(processed_image)
            else:
                # print(f"type of image is {type(image)}, image is {image}")
                raise ValueError("Error processing image")
        return processed_images

    def _prepare_input_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_special_tokens=False, return_dict=True
        )
        mask = [0] * len(conv_ids["input_ids"])
        msg_idx = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template(
                [msg], tokenize=True, add_special_tokens=False
            )
            if isinstance(segment_ids, dict) and "input_ids" in segment_ids:
                seg_len = len(segment_ids["input_ids"])
            else:
                seg_len = len(segment_ids)
            # seg_len = len(segment_ids)
            if msg["role"] == "assistant":
                start = msg_idx + self.prefix_len
                end = msg_idx + seg_len
                mask[start:end] = [1] * (end - start)

            msg_idx += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"]),
        )


class VQADataset(BaseDataset):
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_data = item["images"]
        if not isinstance(image_data, list):
            image_data = [image_data]

        processed_images = self._process_images(image_data)
        messages = self._get_messages(item, len(image_data))
        input_ids, mask, attention_mask = self._prepare_input_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)
        return {
            "images": processed_images,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _get_labels(self, input_ids, mask):
        # print(f"input_ids shape is {input_ids.shape}, mask shape is {mask.shape}")
        # input_ids is a 1d tensor as it is a single sequence
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100
        return labels
