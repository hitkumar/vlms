from math import trunc

import torch
from torch.nn import attention
from torch.nn.modules import padding


class MMStarCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = [i["image"] for i in batch]
        text_data = [i["text_data"] for i in batch]
        answer = [i["answer"] for i in batch]

        images_tensor = torch.stack(images)
        # TODO: Check if padding is correctly added. Normally padding is added to the right.
        encoded_question_sequence = self.tokenizer.batch_encode_plus(
            text_data,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        encoded_answer_sequence = self.tokenizer.batch_encode_plus(
            answer,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        return {
            "image": images_tensor,
            "input_ids": encoded_question_sequence["input_ids"],
            "attention_mask": encoded_question_sequence["attention_mask"],
            "labels": encoded_answer_sequence["input_ids"],
        }


class VQACollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images = [i["image"] for i in batch]
        text_data = [i["text_data"] for i in batch]
        answer = [i["answer"] for i in batch]

        images_tensor = torch.stack(images)
        input_sequences = []
        for i in range(len(text_data)):
            input_sequences.append(f"{text_data[i]} {answer[i]}")

        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            max_length=self.max_length,
            padding_side="left",
            return_tensors="pt",
            truncation=True,
        )
        input_ids = encoded_full_sequences["input_ids"]
        attention_mask = encoded_full_sequences["attention_mask"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone()
        # doesn't get included in loss calculation
        labels[:, -1] = -100

        original_lengths = [len(self.tokenizer.encode(s)) for s in input_sequences]
        for i in range(len(batch)):
            question_length = len(
                self.tokenizer.encode(text_data[i], add_special_tokens=False)
            )

            # Ignore this sample
            if original_lengths[i] > self.max_length:
                labels[i, :] = -100
                continue

            first_token_pos = attention_mask[i].nonzero(as_tuple=True)[0][0].item()
            # first_token_pos is the first position where the attention mask is 1, due to list indexing we are subtracting an extra 1 here because this is the label which is shifted left by 1
            question_end = first_token_pos + question_length - 1
            labels[i, :question_end] = -100

        return {
            "image": images_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
