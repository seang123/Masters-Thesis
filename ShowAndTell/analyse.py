from evaluate import Evaluate


# init the evaluation class
ev = Evaluate("./test_output")

# Generate for an image a predicted caption
img_id, generated, actual = ev.gen_prediction()

print("Image id:", img_id)

act = '\n'.join(actual)
print(f"Predicted:\n{' '.join(generated)}\nActual:\n{act}")

## Save the generated captions
with open(f"./test_output/captions_{img_id}.txt", "w+") as f:
    f.write("Predicted:\n")
    temp = " ".join(generated)
    f.write(temp)
    f.write("\nActual:\n")
    temp = "\n".join(actual)
    f.write(temp)

## Save a .png picture of the image
ev.save_fig(img_id)

print("done.")
