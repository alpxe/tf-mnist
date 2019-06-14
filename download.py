import os
import urllib.request
import sys


# 下载
def maybe_download(file):
    filepath = os.path.join(dest_directory, file)  # 本地文件位置
    url = os.path.join(DEFAULT_SOURCE_URL, file)  # 下载链接

    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (file, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            pass

        respath, _ = urllib.request.urlretrieve(url, filepath, _progress)
        statinfo = os.stat(respath)

        print('Successfully downloaded', file, statinfo.st_size, 'bytes.')
        pass
    pass


dest_directory = "MNIST_DATA"
DEFAULT_SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"  # mnist官方地址

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

if not os.path.exists(dest_directory):  # 如果文件夹不存在
    os.makedirs(dest_directory)  # 创建文件夹
    pass

maybe_download(TRAIN_IMAGES)
maybe_download(TRAIN_LABELS)
maybe_download(TEST_IMAGES)
maybe_download(TEST_LABELS)
