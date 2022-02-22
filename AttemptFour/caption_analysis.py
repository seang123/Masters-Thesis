import pandas as pd
import numpy as np

captions_path = "/fast/seagie/data/captions/"
keys = range(1,73001)


caption_length = []
caption_count = 0
all_captions = []

d = {}

c = 0
for idx, key in enumerate(keys):
    with open(f"{captions_path}/KID{key}.txt") as f:
        content = f.read()
        for cid, i in enumerate(content.splitlines()):
            cap = i.replace(".", " ").replace(",", " ").strip().split(" ")
            cap = [i.lower() for i in cap if i != '']
            cap2 = ['<start>'] + cap + ['<end>']
            caption_length.append( len(cap) )
            caption_count += 1
            all_captions.append(cap)
            d[c] = {'raw':len(cap), 'token':len(cap2)}
            cap = " ".join(cap)
            c += 1

df = pd.DataFrame.from_dict(d, "index")

print("avg. caption length:", sum(caption_length)/caption_count)

print("Min:", min(caption_length))
print("Max:", max(caption_length))
print("Mean:", np.mean(caption_length))
print("----")

print(df.describe(percentiles = [.25, .5, .75, .9, .99]))
