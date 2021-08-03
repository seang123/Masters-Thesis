import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.append('/home/seagie/NSD/Code/Masters-Thesis/')
import utils

img = []

with open("img_indicies.txt") as f:
    lines = f.readlines()
    for line in lines:
        img.append(int(line.rstrip('\n')))

print("total", len(img)) # 30k



img_count = {}

for i in img:
    if i in img_count:
        img_count[i] += 1
    else:
        img_count[i] = 1


print("unique images", len(img_count.keys()))

annt_dict = utils.load_json("../modified_annotations_dictionary.json")

annotations = {}
for k,v in img_count.items():
    if k not in annotations:
        annotations[k] = annt_dict[str(k)]
    else:
        pass

lengths = []
for k, v in annotations.items():
    lengths.append(len(v))


print(set(lengths))


def plot_img_occur(img_count: dict):

    od = dict(sorted(img_count.items()))

    small_count = {}
    top_n = 500
    M = len(img_count) - top_n
    c = 0
    for k, v in img_count.items():
        if c >= M:
            small_count[k] = v
            c += 1
        else:
            c += 1

    fig = plt.figure()
    plt.bar(small_count.keys(), small_count.values())
    plt.xlabel("img")
    plt.ylabel("count")
    plt.title("img occurance subj02")
    plt.savefig("img_occur.png")
    plt.show()
    plt.close(fig)



