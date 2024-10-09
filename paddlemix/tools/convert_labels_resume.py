import os
import sys
import json
from tqdm import tqdm

def convert_labels(in_path, out_dir, split_ratio):
    data_train = []
    data_val = []
    cnt = 0
    styles = []
    max_len = 0
    for line in tqdm(open(in_path, "r")):
        len_attr = len(line.strip().split("\t"))
        if len_attr == 2:
            im_base64, cap_cn = line.strip().split("\t")
        elif len_attr == 3:
            _, im_base64, cap_cn = line.strip().split("\t")
        else:
            print("split length wrong")
            continue

        data_dict = {
            "id": "identity_%s" % str(cnt),
             "conversations": [
                [
                    "请总结图片的图片风格并描述图片内容。其中图片风格例如：写实风格、图标风格、像素风格、水彩风格、卡通风格、插画风格、黑白简笔风格、中国风风格、纯文字风格、艺术风格、素描风格、3D风格、科技风格等。图片内容包括：图片中各个主体的特征、主体之间关系和图片背景。注意描述内容需要具有准确性、连贯性和简洁性。因此该图片的图片风格和描述内容为：",
                    cap_cn,
                ],
            ],
            "image": im_base64
        }
        style = cap_cn.split('，')[0]
        if '风格' in style:
            styles.append(style)
        if '风格' in cap_cn:
            if cnt % split_ratio == 0:
                data_val.append(data_dict)
            data_train.append(data_dict)
            cnt += 1
            if len(cap_cn) > max_len:
                max_len = len(cap_cn)
    styles = set(styles)
    print(len(styles))
    print(styles)
    print(max_len)
    print("num of training set: %d" % len(data_train))
    print("num of validation set: %d" % len(data_val))
    os.makedirs(out_dir, exist_ok=True)
    cp_cmd = "cp chat_template.json %s/" % out_dir
    os.system(cp_cmd)

    out_json = os.path.join(out_dir, "train_resume.json")
    with open(out_json, "w") as file:
        json.dump(data_train[800000:], file, indent=4)

    out_json = os.path.join(out_dir, "val.json")
    with open(out_json, "w") as file:
        json.dump(data_val, file, indent=4)
        

if __name__ == "__main__":
    in_path = sys.argv[1]
    out_dir = sys.argv[2]
    split_ratio = 20

    convert_labels(in_path, out_dir, split_ratio)


