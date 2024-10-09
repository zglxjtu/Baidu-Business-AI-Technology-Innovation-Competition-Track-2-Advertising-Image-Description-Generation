# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle

class CLIPCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data_list):
        if isinstance(data_list[0], dict):
            images = [sample["image"] for sample in data_list]
            text = [sample["text_input"] for sample in data_list]
            batch = self.processor(
                images=images,
                text=text,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="train",
                padding_zero=True,
            )
            return batch
        else:
            images = [sample[0] for sample in data_list]
            labels = [sample[1] for sample in data_list]
            batch = self.processor(
                images=images,
                text=None,
                max_length=77,
                return_tensors="pd",
                return_attention_mask=False,
                mode="eval",
                do_resize=True,
                do_crop=True,
                padding_zero=True,
            )
            batch["labels"] = paddle.to_tensor(np.array(labels))
            return batch


class EVA02Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample[0] for sample in data_list]
        # get labels from teacher's clip_features
        batch = self.processor(
            images=images,
            return_tensors="pd",
            mode=self.mode,
        )
        return batch


class MiniGPT4Collator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        target_texts = [sample["text_input"] for sample in data_list]
        # random text from text_list read by processor and combine it with default prompt
        batch_data = self.processor(images=images, mode="train")
        target_outputs = self.processor.process_target_texts(target_texts)
        batch_data.update(target_outputs)
        return batch_data


class QwenVLCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        input_ids = []
        labels = []
        images = []
        IGNORE_TOKEN_ID = -100
        for record in data_list:

            if isinstance(record, dict) and "input_ids" in record.keys():
                raw_data = record
            else:
                raw_data = self.processor(query=record, mode=self.mode)

            raw_data["input_ids"] += [self.processor.tokenizer.pad_token_id] * (
                self.processor.max_len - len(raw_data["input_ids"])
            )
            raw_data["labels"] += [IGNORE_TOKEN_ID] * (self.processor.max_len - len(raw_data["labels"]))
            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)
                    #raw_data["images"] = paddle.stack(x=self.processor.image_processor(raw_data["images"]), axis=0)

                images.append(raw_data["images"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")
        labels = paddle.to_tensor(data=labels, dtype="int32")
        attention_mask = input_ids.not_equal(y=paddle.to_tensor(self.processor.tokenizer.pad_token_id, dtype="int32"))

        if len(images) > 0:
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(
            input_ids=input_ids,
            labels=labels,
            images=images if 0 < len(images) else None,
            attention_mask=attention_mask,
        )

        return batch_data


class VisualglmCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test", max_seq_length=2048):
        self.processor = processor
        self.mode = mode
        self.max_seq_length = max_seq_length

    def __call__(self, data_list):

        input_ids = []
        labels = []
        images = []

        for record in data_list:
            if "input_ids" not in record.keys():
                raw_data = self.processor(record=record, mode=self.mode)
            else:
                raw_data = record

            pad_len = self.max_seq_length - len(raw_data["input_ids"])
            raw_data["input_ids"] = raw_data["input_ids"] + [self.processor.tokenizer.pad_token_id] * pad_len
            raw_data["labels"] = raw_data["labels"] + [self.processor.tokenizer.pad_token_id] * pad_len
            raw_data["labels"] = [
                (l if l != self.processor.tokenizer.pad_token_id else -100) for l in raw_data["labels"]
            ]

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)
                images.append(raw_data["images"])

            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int64")
        labels = paddle.to_tensor(data=labels, dtype="int64")

        if 0 < len(images):
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(input_ids=input_ids, labels=labels, images=images if 0 < len(images) else None)
        return batch_data


class LLaVACollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test", mixtokens=False):
        self.processor = processor
        self.mode = mode
        self.mixtokens = mixtokens

    def __call__(self, data_list):
        IGNORE_INDEX = -100
        input_ids = []
        labels = []
        images = []
        for record in data_list:

            if isinstance(record, dict) and "input_ids" in record.keys():
                raw_data = record
            else:
                raw_data = self.processor(record=record, mode=self.mode)

            raw_data["input_ids"] += [self.processor.tokenizer.pad_token_id] * (
                self.processor.max_len - len(raw_data["input_ids"])
            )
            raw_data["labels"] += [IGNORE_INDEX] * (self.processor.max_len - len(raw_data["labels"]))

            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)

                images.append(raw_data["images"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")
        labels = paddle.to_tensor(data=labels, dtype="int32")
        attention_mask = input_ids.not_equal(y=paddle.to_tensor(self.processor.tokenizer.pad_token_id, dtype="int32"))

        if len(images) > 0:
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(
            input_ids=input_ids,
            labels=labels,
            images=images if len(images) > 0 else None,
            attention_mask=attention_mask,
        )

        return batch_data


class InternLMXComposer2Collator:
    """Collate examples for InternLMXComposer2Collator"""

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, instances):

        instances = [self.processor(query=instance, mode=self.mode) for instance in instances]

        input_tokens, input_text = tuple(
            [instance[key] for instance in instances] for key in ("input_tokens", "input_text")
        )
        batch = dict(
            input_tokens=input_tokens,
            input_text=input_text,
        )
        if "images" in instances[0].keys():
            input_images = tuple([instance["images"] for instance in instances])
            batch["images"] = input_images

        return dict(samples=batch)

from typing import Iterable, List, Tuple, Union
from paddlenlp.generation import LogitsProcessor
from paddlenlp.transformers import PretrainedTokenizer

def make_context(
    tokenizer: PretrainedTokenizer,
    query: str,
    answer: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")["input_ids"]

        def _tokenize_str(role, content):
            return (
                f"{role}\n{content}",
                tokenizer.encode(role, allowed_special=set(tokenizer.IMAGE_ST))["input_ids"]
                + nl_tokens
                + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))["input_ids"],
            )

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []
        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"
            current_context_size = len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break
        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")["input_ids"]
            + nl_tokens
            + tokenizer.encode(answer, allowed_special=set(tokenizer.IMAGE_ST))["input_ids"]
            + im_end_tokens
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n{answer}{im_end}\n"
    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)["input_ids"]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return raw_text, context_tokens

class MGenVLCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.
    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="test"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        input_ids = []
        labels = []
        images = []
        IGNORE_TOKEN_ID = -100
        tokenizer = self.processor.tokenizer
        for record in data_list:
            prompt = record['conversations'].split('<|im_start|>user\n')[-1].split('<|im_end|>\n<|im_start|>')[0]
            ambrose_record = [
                {"image": record['image']},
                {"text": prompt},
            ]
            ambrose_answer = record['conversations'].split('assistant\n')[-1].split('<|im_end|>\n')[0]
            ambrose_record = tokenizer.from_list_format(ambrose_record)

            raw_text, context_tokens = make_context(
                tokenizer,
                ambrose_record,
                ambrose_answer,
                history=None,
                system="You are a helpful assistant.",
            )

            if isinstance(record, dict) and "input_ids" in record.keys():
                raw_data = record
            else:
                raw_data = self.processor(query=record, mode=self.mode)

            raw_data["input_ids"] = context_tokens
            raw_data["input_ids"] += [self.processor.tokenizer.pad_token_id] * (
                self.processor.max_len - len(raw_data["input_ids"])
            )

            ambrose_label = context_tokens.copy()
            ambrose_label[1:9] = [-100] * (9-1)
            ambrose_label[12:384] = [-100] * (384 - 12)
            ambrose_label[387:389] = [-100] * (389 - 387)
            raw_data["labels"] = ambrose_label

            raw_data["labels"] += [IGNORE_TOKEN_ID] * (self.processor.max_len - len(raw_data["labels"]))

            input_ids.append(raw_data["input_ids"])
            labels.append(raw_data["labels"])

            if "images" in raw_data:
                if isinstance(raw_data["images"], list):
                    raw_data["images"] = paddle.stack(x=raw_data["images"], axis=0)
                    #raw_data["images"] = paddle.stack(x=self.processor.image_processor(raw_data["images"]), axis=0)

                images.append(raw_data["images"])

        input_ids = paddle.to_tensor(data=input_ids, dtype="int32")
        labels = paddle.to_tensor(data=labels, dtype="int32")
        attention_mask = input_ids.not_equal(y=paddle.to_tensor(self.processor.tokenizer.pad_token_id, dtype="int32"))

        if len(images) > 0:
            images = paddle.concat(images, axis=0)
            image_shape = [-1, 3] + images.shape[-2:]
            images = images.reshape(image_shape)

        batch_data = dict(
            input_ids=input_ids,
            labels=labels,
            images=images if 0 < len(images) else None,
            attention_mask=attention_mask,
        )

        return batch_data
