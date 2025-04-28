# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
from util import augment,visualize_linear_params, pad_imgs
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 对所有图片做增强
train_imgs_aug_only = np.array([augment(img) for img in train_imgs])
train_imgs_aug = np.concatenate([train_imgs, train_imgs_aug_only], axis=0)
train_labs_aug = np.concatenate([train_labs, train_labs], axis=0)

train_imgs = pad_imgs(train_imgs)
valid_imgs = pad_imgs(valid_imgs)
train_imgs_aug = pad_imgs(train_imgs_aug)


linear_model = nn.models.Model_MLP([train_imgs_aug.shape[-1], 60, 10], 'ReLU', [1e-4, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 100, 10], 'ReLU', [1e-4, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 60, 30, 10], 'ReLU', [1e-4, 1e-4, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 60, 10], 'ReLU', [0.1, 1e-4])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 60, 10], 'ReLU', [1e-4, 0.1])
# linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 60, 10], 'ReLU', [0.1, 0.1])
lenet = nn.models.LeNet5()

model = linear_model

optimizer = nn.optimizer.Adam(init_lr=0.06, model=model, beta1=0.9, beta2=0.99)
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=linear_model)
# optimizer = nn.optimizer.SGD_Momentum(init_lr=0.06, model=linear_model, momentum=0.97)

scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs_aug.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs_aug, train_labs_aug], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()