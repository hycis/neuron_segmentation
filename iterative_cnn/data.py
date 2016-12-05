
from PIL import Image
import numpy as np
import tensorgraph as tg


class Iris(object):


    def __init__(self, patch_size=(32,32), min_pos_ratio=0.2, num_patch_per_img=200,
                 train_valid = [5,1], batchsize=32, shuffle=True):
        self.patch_size = patch_size
        self.min_pos_ratio = min_pos_ratio
        self.num_patch_per_img = num_patch_per_img
        self.data_dir = '../data/manual'
        self.train_valid = train_valid
        self.batchsize = batchsize
        self.shuffle = shuffle


    def make_data(self):
        test_num = ['01', '02', '05', '06', '09']
        train_num = ['21', '22', '25', '26', '29']
        test_labels = ['{}/{}_manual1.gif'.format(self.data_dir,num) for num in test_num]
        test_images = ['{}/{}_test.tif'.format(self.data_dir,num) for num in test_num]
        labels_path = ['{}/{}_manual1.gif'.format(self.data_dir,num) for num in train_num] + test_labels
        images_path = ['{}/{}_training.tif'.format(self.data_dir,num) for num in train_num] + test_images
        images_num = train_num + test_num
        labels = []
        images = []
        for img_path, lbl_path in zip(images_path, labels_path):
            lbl = Image.open(lbl_path)
            lbl = np.asarray(lbl)[:,:,np.newaxis]
            lbl = lbl / 255
            labels.append(lbl)

            im = Image.open(img_path)
            im = np.asarray(im) / 255
            images.append(im)

        img_patches = []
        lbl_patches = []
        for img, lbl, num in zip(images, labels, images_num):
            print '..processing img {}'.format(num)
            count = 0
            h, w, c = lbl.shape
            while count < self.num_patch_per_img:
                y = np.random.randint(0, h-self.patch_size[0])
                x = np.random.randint(0, w-self.patch_size[1])
                lbl_crop = lbl[y:y+self.patch_size[0], x:x+self.patch_size[1], :]
                if lbl_crop.mean() > self.min_pos_ratio:
                    lbl_patches.append(lbl_crop)
                    img_crop = img[y:y+self.patch_size[0], x:x+self.patch_size[1], :]
                    img_patches.append(img_crop)
                    count += 1

        img_patches = np.asarray(img_patches)
        lbl_patches = np.asarray(lbl_patches)

        if self.shuffle:
            print '..shuffling'
            np.random.seed(1012)
            shf_idx = np.arange(len(img_patches))
            np.random.shuffle(shf_idx)
            img_patches = img_patches[shf_idx]
            lbl_patches = lbl_patches[shf_idx]

        num_train = float(self.train_valid[0]) / sum(self.train_valid) * len(img_patches)
        num_train = int(num_train)
        train = tg.SequentialIterator(img_patches[:num_train], lbl_patches[:num_train], batchsize=self.batchsize)
        valid = tg.SequentialIterator(img_patches[num_train:], lbl_patches[num_train:], batchsize=self.batchsize)
        return train, valid


# class Skin(object):




if __name__ == '__main__':
    data = Iris()
    data.make_data()
