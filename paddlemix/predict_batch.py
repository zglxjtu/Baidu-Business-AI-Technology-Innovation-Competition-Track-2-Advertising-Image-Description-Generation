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

os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'
os.environ['inplace_normalize'] = '1'
os.environ['fuse_relu_before_depthwise_conv'] = '1'

seed = 24
paddle.seed(seed)
random.seed(seed)
np.random.seed(seed)
dtype = "bfloat16"
# dtype = "float16"
if not paddle.amp.is_bfloat16_supported():
    print("bfloat16 is not supported on your device,change to float16")
    dtype = "float16"

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
model_name_or_path = sys.argv[3]

tokenizer = AutoTokenizerMIX.from_pretrained(model_name_or_path)
processor, _ = AutoProcessorMIX.from_pretrained(model_name_or_path)
model_config = AutoConfigMIX.from_pretrained(model_name_or_path, dtype=dtype)
model = AutoModelMIX.from_pretrained(model_name_or_path, config=model_config, dtype=dtype)
model.eval()
s_time = time.time()
f_out = open(output_file_path, "w")

print("------predicting starts------")
prompt = "请总结图片的图片风格并描述图片内容。其中图片风格例如：写实风格、图标风格、像素风格、水彩风格、卡通风格、插画风格、黑白简笔风格、中国风风格、纯文字风格、艺术风格、素描风格、3D风格、科技风格等。图片内容包括：图片中各个主体的特征、主体之间关系和图片背景。注意描述内容需要具有准确性、连贯性和简洁性。因此该图片的图片风格和描述内容为："
i = 0
query = []
inputs_images = []
response_list = []
im_id_list = []
thred = 10
for line in open(input_file_path, "r"):
    len_attr = len(line.strip().split("\t"))
    if len_attr == 2:
        im_id, im_base64 = line.strip().split("\t")
    elif len_attr == 3:
        im_id, im_base64, caption = line.strip().split("\t")
    else:
        print("split length wrong")
        continue

    query1 = [
        {"image": im_base64},
        {"text": prompt},
    ]
    input = processor(query=query1, return_tensors="pd")
    query1 = tokenizer.from_list_format(query1)
    query.append(query1)
    inputs_images.append(input["images"])
    i += 1
    if not i % thred:
        inputs_images = paddle.concat(inputs_images, axis=0)
        response = model.chat_batch(tokenizer, query=query, history=None, images=inputs_images)
        query = []
        inputs_images = []
        response_list = response_list + response
    im_id_list.append(im_id)
    # f_out.write("%s\t%s\n" % (im_id, response))
if i % thred:
    inputs_images = paddle.concat(inputs_images, axis=0)
    response = model.chat_batch(tokenizer, query=query, history=None, images=inputs_images)
    response_list = response_list + response
for im_id, response in zip(im_id_list, response_list):
    print(response)
    f_out.write("%s\t%s\n" % (im_id, response))
print("------predicting finished-----", time.time()-s_time)
