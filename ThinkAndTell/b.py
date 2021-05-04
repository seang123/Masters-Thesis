import tensorflow as tf

"""
ds will get shuffled once but be consistence across epochs if reshuffle_each_iter=False

"""

ds = tf.data.Dataset.range(50)

ds = ds.map(lambda x: x + 1)

ds = ds.shuffle(50, reshuffle_each_iteration=False)

ds_train = ds.take(40)
ds_test = ds.skip(40)#.shuffle(10)

with tf.device('/cpu'):
    for i in range(0, 2):
        print("--------train-------")
        for batch, v in ds_train.enumerate():
            print(batch.numpy(), "-",  v.numpy())
        print("--------test--------")
        for batch, v in ds_test.enumerate():
            print(batch.numpy(), "-", v.numpy())
