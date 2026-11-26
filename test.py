from huggingface_hub import snapshot_download
import glob, gzip, string
import numpy as np
from datetime import datetime

start = datetime.now().timestamp()

path = snapshot_download(repo_id="Heliosoph/EMNIST", repo_type="dataset", cache_dir="./training_data/")

def find_file(name: str):
    return glob.glob(f"{path}/**/{name}", recursive=True)[0]

def load_images(file_name):
    with gzip.open(find_file(file_name), "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).transpose(0, 2, 1) / 255.0

def load_labels(file_name):
    with gzip.open(find_file(file_name), "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=8)

x_train = load_images("emnist-byclass-train-images-idx3-ubyte.gz")
y_train = load_labels("emnist-byclass-train-labels-idx1-ubyte.gz")
x_test  = load_images("emnist-byclass-test-images-idx3-ubyte.gz")
y_test  = load_labels("emnist-byclass-test-labels-idx1-ubyte.gz")

classes = string.digits + string.ascii_uppercase + string.ascii_lowercase

print(x_train[0])
print("Label:", classes[y_train[0]])