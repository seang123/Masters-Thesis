import my_utils as uu

annt_dict = uu.load_json("../../modified_annotations_dictionary.json")


print(annt_dict[str(15597)])

keys = []

with open('./train_keys.txt', 'r') as f:
    lines = f.read()
    lines = lines.splitlines()

    for l in lines:
        keys.append(l)


#dup = []
#for i in range(0, len(keys)):
#    for j in range(0, len(keys)):
#        if j == i:
#            continue
#        else:
#            if keys[i] == keys[j]:
#                dup.append(keys[i])
#print(len(keys), len(dup))

print(len(keys))
print(len(list(set(keys))))


