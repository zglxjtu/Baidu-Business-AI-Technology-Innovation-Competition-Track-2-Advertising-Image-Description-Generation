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
                if file not in ["config.json", "generation_config.json", "lora_model_state.pdparams", "optimizer.pdopt"]:
                    src_path = os.path.join(src_dir, file)
                    cp_cmd = "cp -a %s %s" % (src_path, des_dir)
                    print(cp_cmd)
                    os.system(cp_cmd)


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    lora_config = LoRAConfig.from_pretrained(args.lora_path)
    dtype = lora_config.dtype
    lora_config.merge_weights = True

    # Load model config
    model_config = AutoConfigMIX.from_pretrained(args.model_name_or_path, dtype=dtype)

    # Load model
    model = AutoModelMIX.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        dtype=dtype,
    )

    model = LoRAModel.from_pretrained(model=model, lora_path=args.lora_path, lora_config=lora_config)
    model.eval()
    if args.merge_model_path is None:
        args.merge_model_path = args.lora_path

    model_state_dict = model.model.state_dict()
    for key in list(model_state_dict):
        if "lora" in key:
            del model_state_dict[key]
    model.model.save_pretrained(args.merge_model_path, state_dict=model_state_dict)
    #copy other configs
    cp_config(args.lora_path, args.merge_model_path)

    print("--------merge lora successfully----------")


if __name__ == "__main__":
    merge()
