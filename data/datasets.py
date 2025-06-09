import models.config as cfg
import torch
from PIL import Image
from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image_data = item["images"]
        if isinstance(image_data, list) and len(image_data) > 0:
            image = image_data[0]
        else:
            image = image_data

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {index}")
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size
            )

        text_data = item["texts"]
        if isinstance(text_data, list) and len(text_data) > 0:
            text = text_data[0]
        else:
            text = text_data

        question = text["user"]
        # This makes the model learn to stop after predicting the answer
        answer = text["assistant"] + self.tokenizer.eos_token
        formatted_text = f"Question: {question} Answer: "

        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer,
        }


class MMStarDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        image = item["image"]

        # Now process the image
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            processed_image = self.image_processor(image)
        else:
            print(f"Error processing image at index {index}")
            processed_image = torch.zeros(
                3, cfg.VLMConfig.vit_img_size, cfg.VLMConfig.vit_img_size
            )

        question = item["question"]
        answer = item["answer"] + self.tokenizer.eos_token
        formatted_text = (
            f"Question: {question} \nAnswer only with the letter! \nAnswer: "
        )
        return {
            "image": processed_image,
            "text_data": formatted_text,
            "answer": answer,
        }
