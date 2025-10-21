import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

class TFDSDataLoader:
    def __init__(self, dataset, split, batch_size, is_training=True,
                transform=None, one_hot=True, augment_method=None, seed=0):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.is_training = is_training
        self.transform = transform
        self.one_hot = one_hot
        self.augment_method = augment_method
        self.seed = seed

        # Load dataset + info
        ds, self.ds_info = tfds.load(
            dataset, split=split, as_supervised=True, with_info=True
        )
        self.num_classes = self.ds_info.features["label"].num_classes

        # Build pipeline
        self.ds = self._build_pipeline(ds)

    def _build_pipeline(self, ds):
        def map_fn(img, label):
            # standardize + normalize, apply transformation if called
            img, label = self._preprocess(img, label) 
            # one hot encoding for softmax
            if self.one_hot: 
                label = tf.one_hot(label, self.num_classes)
            return img, label

        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

        if self.is_training:
            ds = ds.shuffle(10 * self.batch_size, seed=self.seed, reshuffle_each_iteration=False)
            ds = ds.repeat()

        ds = ds.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return tfds.as_numpy(ds)
    
    def _apply_transforms(self, img):
        for transform_fn in self.transform:
            img = transform_fn(img)
        return img

    def _preprocess(self, img, label):
        #normalize to [0,1]
        img = tf.cast(img, tf.float32) / 255.0 

        #standardize img 
        if self.dataset in ["cifar100", "cifar10"]:
            img = tf.image.resize(img, [224, 224])
            mean = tf.constant([0.5071, 0.4865, 0.4409])
            std = tf.constant([0.2673, 0.2564, 0.2762])
        elif self.dataset in ["fashion_mnist", "kmnist", "mnist"]:
            mean = tf.constant([0.1307]) #Grayscale
            std = tf.constant([0.3081])
        img = (img - mean) / std  

        #transformation
        if self.transform:
            img = self._apply_transforms(img)
        return img, label

    def __iter__(self):
        return iter(self.ds)

    def get_info(self):
        return self.ds_info
    
    def plot_image(self, n=8):
        batch = next(iter(self))
        images, labels = batch
        images = images[:n]

        fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
        for i in range(n):
            img = images[i]
            if img.shape[-1] == 1: 
                axes[i].imshow(img.squeeze(), cmap='gray') #for grayscale images
            else:
                axes[i].imshow(img) #RGB images
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

def random_flip(img):
    return tf.image.random_flip_left_right(img)

def rotate(img):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(img, k)


from collections import Counter
import numpy as np

def get_class_distribution(loader):
    counter = Counter()
    batch_count = 0
    for images, labels in loader:
        # Convert one-hot to class index
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        counter.update(labels.tolist())
        batch_count += 1
    print(f"class distribution:")
    for cls, count in sorted(counter.items()):
        print(f"  Class {cls}: {count}")
