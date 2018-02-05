import os
import random
import PIL
import PIL.Image
import numpy as np
import sys


SAMPLES = 1000000
DIM = 5
RESIZE = True
RESIZE_DIM = 300
OUTPUT_FILE = 'quilt_db.npy'

def main(argv):
    imagenet_train_dir = argv[1]

    assert SAMPLES % 1000 == 0

    db = np.zeros((SAMPLES, DIM, DIM, 3), dtype=np.float32)

    idx = 0
    files = []
    for d in os.listdir(imagenet_train_dir):
        d = os.path.join(imagenet_train_dir, d)
        files.extend(os.path.join(d, i) for i in os.listdir(d) if i.endswith('.JPEG'))
    for f in random.sample(files, SAMPLES):
        img = load_image(f)
        h, w, _ = img.shape
        h_start = random.randint(0, h - DIM)
        w_start = random.randint(0, w - DIM)
        crop = img[h_start:h_start+DIM, w_start:w_start+DIM, :]
        db[idx, :, :, :] = crop
        idx += 1
        
        if idx % 100 == 0:
            print('%.2f%% done' % (100 * (float(idx) / SAMPLES)))

    np.save(OUTPUT_FILE, db)


def load_image(path):
    image = PIL.Image.open(path)
    if RESIZE:
        if image.height > image.width:
            image = image.resize((int(float(image.width) / image.height * RESIZE_DIM), RESIZE_DIM))
        elif image.width > image.height:
            image = image.resize((RESIZE_DIM, int(float(image.height) / image.width * RESIZE_DIM)))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img


if __name__ == '__main__':
    main(sys.argv)

