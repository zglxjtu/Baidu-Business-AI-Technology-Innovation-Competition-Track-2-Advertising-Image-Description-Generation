# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import sys

import paddle
import time
import random
import numpy as np

from PIL import Image
from io import BytesIO
import base64

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

from auto import (
    AutoConfigMIX,
    AutoModelMIX,
    AutoProcessorMIX,
    AutoTokenizerMIX,
)
from utils.log import logger

seed = 24
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)
dtype = "bfloat16"
#dtype = "float16"
if not paddle.amp.is_bfloat16_supported():
    logger.warning("bfloat16 is not supported on your device,change to float16")
    dtype = "float16"

model_name_or_path = sys.argv[1]
tokenizer = AutoTokenizerMIX.from_pretrained(model_name_or_path)
processor, _ = AutoProcessorMIX.from_pretrained(model_name_or_path)
model_config = AutoConfigMIX.from_pretrained(model_name_or_path, dtype=dtype)
model = AutoModelMIX.from_pretrained(model_name_or_path, config=model_config, dtype=dtype)
model.eval()

prompt = "请描述图片内容"
start = time.time()
query1 = [
    {"image": "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"},
    {"text": prompt},
]
input = processor(query=query1, return_tensors="pd")
query1 = tokenizer.from_list_format(query1)
response, history = model.chat(tokenizer, query=query1, history=None, images=input["images"])
response = response.replace("\n", " ").replace("\r", " ")
print("prompt: %s" % prompt)
print("response: %s" % response)
print("------------------")
    
end = time.time()
length = end - start
print("It took", length, "seconds!")