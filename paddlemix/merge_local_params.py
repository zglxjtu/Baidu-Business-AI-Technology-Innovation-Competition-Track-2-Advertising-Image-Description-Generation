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
import os
import argparse

import paddle
from paddlenlp.peft import LoRAConfig, LoRAModel

from auto import AutoConfigMIX, AutoModelMIX


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of pretrained model.")
    parser.add_argument(
        "--lora_path", default=None, required=True, help="The directory of LoRA parameters. Default to None"
    )
    parser.add_argument("--merge_model_path", default=None, help="The directory of merged parameters. Default to None")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    return parser.parse_args()

def cp_config(src_dir, des_dir):
    for (root, _, files) in os.walk(src_dir):
        if root == src_dir:
            for file in files:
                if file not in ["config.json", "generation_config.json", "model_state.pdparams", "optimizer.pdopt"]:
                    src_path = os.path.join(src_dir, file)
                    cp_cmd = "cp -a %s %s" % (src_path, des_dir)
                    print(cp_cmd)
                    os.system(cp_cmd)


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    dtype = "float16"

    # Load model config
    model_config = AutoConfigMIX.from_pretrained(args.model_name_or_path, dtype=dtype)

    # Load model
    model = AutoModelMIX.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        dtype=dtype,
    )

    model_state_dict = model.state_dict()

    local_model = AutoModelMIX.from_pretrained(
        args.lora_path,
        config=model_config,
        dtype=dtype,
    )
    local_state_dict = local_model.state_dict()

    trainable_list = ["transformer.llm.h.31", "transformer.llm.h.30", "transformer.llm.h.29", "transformer.llm.h.28",
                      "transformer.llm.h.27", "transformer.llm.h.26", "transformer.llm.h.25", "transformer.llm.h.24",
                      "transformer.llm.h.23", "transformer.llm.h.22", "transformer.llm.h.21", "transformer.llm.h.20",
                      "transformer.llm.h.19", "transformer.llm.h.18", "transformer.llm.h.17", "transformer.llm.h.16"]
    sub_list = [".attn.c_attn", ".attn.c_proj", ".mlp.w1", ".mlp.w2"]

    ambrose_trainable_list = []
    for a in trainable_list:
        for b in sub_list:
            ambrose_trainable_list.append(a + b)
    ambrose_trainable_list.append("visual.attn_pool")
    ambrose_trainable_list.append("visual.proj")
    for name in list(model_state_dict):
        if any(nd in name for nd in ambrose_trainable_list):
            model_state_dict[name] = local_state_dict[name]

    del local_state_dict
    model.save_pretrained(args.merge_model_path, state_dict=model_state_dict)
    cp_config(args.lora_path, args.merge_model_path)

    print("--------merge local successfully----------")


if __name__ == "__main__":
    merge()
