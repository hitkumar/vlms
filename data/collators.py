import torch
import torch.nn.functional as F


class BaseCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_batch(self, batch, max_length):
        """
        batch is a dict of tensors with 4 keys
        - input_ids list[length]
        - attention_mask list[length]
        - labels list[length]
        - image list[3, img_size, img_size]
        We do left padding here.
        """
        pad_token_id = self.tokenizer.pad_token_id
        batch["input_ids"] = [
            F.pad(ids, (max_length - len(ids), 0), value=pad_token_id)
            for ids in batch["input_ids"]
        ]
        batch["attention_mask"] = [
            F.pad(mask, (max_length - len(mask), 0), value=0)
            for mask in batch["attention_mask"]
        ]
        # pad value is -100 to ignore loss calculation
        batch["labels"] = [
            F.pad(labels, (max_length - len(labels), 0), value=-100)
            for labels in batch["labels"]
        ]
        return batch

    def prepare_batch(self, batch, max_length=None):
        # Convert list of dicts to dict of lists
        batch = {key: [b[key] for b in batch] for key in batch[0]}
        if max_length is not None:
            max_len = max_length
            batch = self._discard_elements_greater_than_max_length(batch, max_length)
        else:
            max_len = max(map(len, batch["input_ids"]))

        batch = self._pad_batch(batch, max_len)
        return {
            "input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            # this is a list of images, each image is a list of tensors of shape [3, img_size, img_size]
            "images": batch["images"],
        }

    def _discard_elements_greater_than_max_length(self, batch, max_length):
        filtered_data = [
            (ids, mask, labels, image)
            for ids, mask, labels, image in zip(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                batch["images"],
            )
            if len(ids) <= max_length
        ]
        batch_ids, batch_mask, batch_labels, batch_image = zip(*filtered_data)
        return {
            "input_ids": list(batch_ids),
            "attention_mask": list(batch_mask),
            "labels": list(batch_labels),
            "images": list(batch_image),
        }


class VQACollator(BaseCollator):
    def __init__(self, tokenizer, max_length):
        super().__init__(tokenizer)
        self.max_length = max_length

    def __call__(self, batch):
        batch = self.prepare_batch(batch, self.max_length)
        return batch
