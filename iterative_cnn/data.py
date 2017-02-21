
from PIL import Image
import numpy as np
import tensorgraph as tg
import glob
# import cv2s


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



    def extract_patches(self, images_path, labels_path):
        img_patches = []
        lbl_patches = []
        for img_path, lbl_path in zip(images_path, labels_path):
            lbl = Image.open(lbl_path)
            lbl = np.asarray(lbl)[:,:,np.newaxis]
            lbl = lbl / 255
            img = Image.open(img_path)
            img = np.asarray(img) / 255
            print('..processing img {}'.format(img_path))
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
        return np.asarray(img_patches), np.asarray(lbl_patches)


    def make_data(self):
        test_num = ['01', '02', '05', '06', '09']
        train_num = ['21', '22', '25', '26', '29']
        test_labels_path = ['{}/{}_manual1.gif'.format(self.data_dir,num) for num in test_num]
        test_images_path = ['{}/{}_test.tif'.format(self.data_dir,num) for num in test_num]
        train_labels_path = ['{}/{}_manual1.gif'.format(self.data_dir,num) for num in train_num]
        train_images_path = ['{}/{}_training.tif'.format(self.data_dir,num) for num in train_num]

        train_patches, train_lbl = self.extract_patches(train_images_path, train_labels_path)
        test_patches, test_lbl = self.extract_patches(test_images_path, test_labels_path)

        if self.shuffle:
            print('..shuffling')
            np.random.seed(1012)
            shf_idx = np.arange(len(train_patches))
            np.random.shuffle(shf_idx)
            train_patches = train_patches[shf_idx]
            train_lbl = train_lbl[shf_idx]

        # num_train = float(self.train_valid[0]) / sum(self.train_valid) * len(img_patches)
        # num_train = int(num_train)
        train = tg.SequentialIterator(train_patches, train_lbl, batchsize=self.batchsize)
        valid = tg.SequentialIterator(test_patches, test_lbl, batchsize=self.batchsize)
        return train, valid


class Skin(object):

    def __init__(self, train_valid, batchsize, shuffle=True):
        self.input_dir = '../data/Skin'
        self.train_valid = train_valid
        self.shuffle = shuffle
        self.batchsize = batchsize
        self.img_augment = False
        self.load_data = True


    def make_data(self, img_augment=False):
        images = glob.glob(self.input_dir + '/Image/*.jpg')
        X = []
        y = []
        _IMG_INPUT_DIM_ = (3, 128, 128)
        c, h, w = _IMG_INPUT_DIM_

        if self.load_data:
            print('..loading data')
            # with open() as Xin, open() as yin:
            X = np.load('../data/skin_X.npy')
            y = np.load('../data/skin_y.npy')
        else:
            for imgpath in images:
                X_img = []
                y_img = []
                # import pdb; pdb.set_trace()
                imgx = Image.open(imgpath)
                imgx = imgx.resize((w,h))
                imgx = np.asarray(imgx) / 255.0

                lblpath = imgpath.replace('Image', 'Label')
                lblpath = lblpath.replace('.jpg', '_Segmentation.png')

                # ddir = os.path.dirname(imgpath)
                # ddir = os.path.dirname(ddir)
                # lbldir = ddir + '/Label'
                # fname = os.path.basename(imgpath)
                # fname = fname.replace('.jpg', '_Segmentation.png')
                # fname = fname.replace('.' + args.extension, '')

                # lblpath = lbldir + '/' + args.label_pattern.replace('%', fname)
                # lblpath = lbldir + '/' + fname
                # imgy = cv2.imread(lblpath)
                imgy = Image.open(lblpath)
                imgy = imgy.resize((w,h))
                imgy = np.asarray(imgy)[:,:,np.newaxis] / 255.0
                # import pdb; pdb.set_trace()
                # imgx = cv2.resize(imgx, (w,h))
                # imgy = cv2.resize(imgy, (w,h))
                X_img.append(imgx)
                y_img.append(imgy)


                if self.img_augment:
                    imh, imw, imc = imgx.shape

                    # scales = [0.5, 0.6, 0.7, 0.8]
                    # scales = [0.5, 0.8]
                    # pts2s = []
                    # for scale in scales:
                    #     pts21 = np.float32([[0,0],[scale*imw,imh],[imw,0],[imw,imh]])
                    #     pts22 = np.float32([[0,0],[0,scale*imh],[imw,0],[imw,imh]])
                    #     pts23 = np.float32([[0,0],[0,imh],[scale*imw,0],[imw,imh]])
                    #     pts24 = np.float32([[0,0],[0,imh],[imw,scale*imh],[imw,imh]])
                    #     pts25 = np.float32([[0,0],[0,imh],[imw,0],[scale*imw,imh]])
                    #     pts26 = np.float32([[0,0],[0,imh],[imw,0],[imw,scale*imh]])
                    #     pts27 = np.float32([[scale*imw,0],[scale*imw,imh],[imw,0],[imw,imh]])
                    #     pts28 = np.float32([[0,scale*imh],[scale*imw,imh],[imw,0],[imw,imh]])
                    #     pts2s += [pts21, pts22, pts23, pts24, pts25, pts26, pts27, pts28]
                    #
                    #
                    # pts1 = np.float32([[0,0],[0,imh],[imw,0],[imw,imh]])
                    # for pts2 in pts2s:
                    #     m = cv2.getPerspectiveTransform(pts1,pts2)
                    #     dstx = cv2.warpPerspective(imgx,m,(imw,imh))
                    #     dsty = cv2.warpPerspective(imgy,m,(imw,imh))
                    #     dstx = cv2.resize(dstx, (w,h))
                    #     dsty = cv2.resize(dsty, (w,h))
                    #     X_img.append(dstx)
                    #     y_img.append(dsty)

                    rotx = []
                    roty = []
                    for imgx, imgy in zip(X_img, y_img):
                        for angle in [90, 180, 270]:
                            m = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
                            dstx = cv2.warpAffine(imgx, m, (w,h))
                            dsty = cv2.warpAffine(imgy, m, (w,h))
                            rotx.append(dstx)
                            roty.append(dsty)

                    X_img += rotx
                    y_img += roty


                X += X_img
                y += y_img
                print('lenX', len(X))
                print('leny', len(y))


            X = np.asarray(X)
            # # X = np.rollaxis(X, 3, 1)
            y = np.asarray(y)
            # # y = np.rollaxis(y, 3, 1)
            # # import pdb; pdb.set_trace()
            #
            with open('../data/skin_X.npy', 'wb') as Xout,\
                open('../data/skin_y.npy', 'wb') as yout:
                np.save(Xout, X)
                np.save(yout, y)




        num_train = float(self.train_valid[0]) / sum(self.train_valid) * len(X)
        num_train = int(num_train)

        if self.shuffle:
            print('..shuffling')
            np.random.seed(1012)
            shf_idx = np.arange(len(X))
            np.random.shuffle(shf_idx)
            X = X[shf_idx]
            y = y[shf_idx]

        train = tg.SequentialIterator(X[:num_train], y[:num_train], batchsize=self.batchsize)
        valid = tg.SequentialIterator(X[num_train:], y[num_train:], batchsize=self.batchsize)
        return train, valid



class DataBlks(object):

    def __init__(self, paths, depth, height, width, batchsize, min_density, num_patch_per_img, rotate=False):
        self.paths = paths
        self.depth = depth
        self.height = height
        self.width = width
        self.batchsize = batchsize
        self.min_density = min_density
        self.num_patch_per_img = num_patch_per_img
        self.rotate = rotate

    # def __iter__(self):
        # self.path_iter = iter(self.paths)
        # return self


    # def __next__(self):
    #     X_path, y_path = next(self.path_iter)
    #     print('..loading data blk')
    #     with open(X_path) as Xin, open(y_path) as yin:
    #         X_npy = np.expand_dims(np.load(Xin), -1)
    #         X_npy /= 255
    #         y_npy = np.expand_dims(np.load(yin), -1)
    #         y_npy /= 100
    #         print('X_npy max', np.max(X_npy))
    #         print('y_npy max', np.max(y_npy))
    #         print('X.shape:', X_npy.shape)
    #         print('y.shape:', y_npy.shape)
    #     print('..extracting patches')
    #     X_patches, y_patches = self.extract_patches(X_npy, y_npy)
    #     del X_npy, y_npy
    #     blk = tg.SequentialIterator(X_patches, y_patches, batchsize=self.batchsize)
    #     return blk

    def make_data(self):
        X_patches = []
        y_patches = []
        print('creating patches')
        for X_path, y_path in self.paths:
            with open(X_path) as Xin, open(y_path) as yin:
                X_npy = np.expand_dims(np.load(Xin), -1)

                print('before shrinked:', X_npy.shape)
                x_f = 0
                x_b = 0
                y_f = 0
                y_b = 0
                z_f = 0
                z_b = 0
                while True:
                    shrinked = False
                    if X_npy[:10, :, :, :].sum() == 0:
                        X_npy =  X_npy[10:, :, :, :]
                        shrinked = True
                        x_f += 10
                    if X_npy[-10:, :, :, :].sum() == 0:
                        X_npy = X_npy[:-10, :, :, :]
                        shrinked = True
                        x_b += 10

                    if X_npy[:, :10, :, :].sum() == 0:
                        X_npy =  X_npy[:, 10:, :, :]
                        shrinked = True
                        y_f += 10
                    if X_npy[:, -10:, :, :].sum() == 0:
                        X_npy = X_npy[:, :-10, :, :]
                        shrinked = True
                        y_b += 10

                    if X_npy[:, :, :10, :].sum() == 0:
                        X_npy =  X_npy[:, :, 10:, :]
                        shrinked = True
                        z_f += 10
                    if X_npy[:, :, -10:, :].sum() == 0:
                        X_npy = X_npy[:, :, :-10, :]
                        shrinked = True
                        z_b += 10

                    if not shrinked:
                        break
                    # print('after shrinked:', X_npy.shape)

                print('after shrinked:', X_npy.shape)


                # import pdb; pdb.set_trace()
                # print('X unique', np.unique(X_npy))
                X_npy = X_npy / 255.0
                y_npy = np.expand_dims(np.load(yin), -1)

                    # y_npy = y_npy[x_f:-x_b, y_f:-y_b, z_f:-z_b, :]


                y_npy = y_npy[x_f:, y_f:, z_f:, :]
                if x_b > 0:
                    y_npy = y_npy[:-x_b, :, :, :]
                if y_b > 0:
                    y_npy = y_npy[:, :-y_b, :, :]
                if z_b > 0:
                    y_npy = y_npy[:, :, :-z_b, :]


                print('y_npy after shrinked:', y_npy.shape)
                # print('y unique', np.unique(y_npy))
                # import pdb; pdb.set_trace()
                y_npy /= 100
            X_patch, y_patch = self.extract_patches(X_npy, y_npy)
            del X_npy, y_npy
            X_patches.append(X_patch)
            y_patches.append(y_patch)
            print("{} done! of {}".format(X_path, len(self.paths)))
        X_patches = np.concatenate(X_patches)
        y_patches = np.concatenate(y_patches)

        ridx = np.arange(len(X_patches))
        np.random.shuffle(ridx)
        X_patches = X_patches[ridx]
        y_patches = y_patches[ridx]
        print('X shape', X_patches.shape)
        print('y shape', y_patches.shape)
        return tg.SequentialIterator(X_patches, y_patches, batchsize=self.batchsize)



    def extract_patches(self, X_npy, y_npy):
        img_patches = []
        lbl_patches = []
        count = 0
        d, h, w, c = X_npy.shape
        density = X_npy.mean()
        print('density:', density)
        # import pdb; pdb.set_trace()
        num_patch_per_img = np.prod(X_npy.shape) / (self.height*self.width*self.depth)
        print('number patches before:', num_patch_per_img)
        # num_patch_per_img = 1000 if num_patch_per_img > 1000 else num_patch_per_img
        num_patch_per_img /= 100
        num_patch_per_img = num_patch_per_img * density * 200
        print('number patches picked:', num_patch_per_img)
        ttl_num_patches_tried = 0
        while count < num_patch_per_img:
            y = np.random.randint(0, h-self.height)
            x = np.random.randint(0, w-self.width)
            z = np.random.randint(0, d-self.depth)

            lbl_crop = y_npy[z:z+self.depth, y:y+self.height, x:x+self.width, :]
            # if lbl_crop.mean() > 0:
            # if np.isnan(lbl_crop.mean()):
            #     import pdb; pdb.set_trace()

            if lbl_crop.mean() > self.min_density:
                # print('patch mean:', lbl_crop.mean())
                # import pdb; pdb.set_trace()
                img_crop = X_npy[z:z+self.depth, y:y+self.height, x:x+self.width, :]
                if self.rotate:
                    import pdb; pdb.set_trace()
                    lbl_patches += [lbl[:,:,:,np.newaxis] for lbl in rotations12(lbl_crop[:,:,:,0])][:6]
                    img_patches += [img[:,:,:,np.newaxis] for img in rotations12(img_crop[:,:,:,0])][:6]
                else:
                    lbl_patches.append(lbl_crop)
                    img_patches.append(img_crop)
                count += 1
                # print count
            ttl_num_patches_tried += 1
            if ttl_num_patches_tried > 4*num_patch_per_img:
                break
        print('number of patches tried:', ttl_num_patches_tried)

        img_patches = np.asarray(img_patches)
        lbl_patches = np.asarray(lbl_patches)
        # import pdb; pdb.set_trace()
        return img_patches, lbl_patches


    def next(self):
        return self.__next__()

    def __len__(self):
        return self.num_patch_per_img * len(self.paths)


def rotations12(polycube):
    for i in range(3):
        polycube = np.transpose(polycube, (1, 2, 0))
        for angle in range(4):
            polycube = np.rot90(polycube)
            yield polycube



def datablks(d, h, w, batchsize, min_density, num_patch_per_img=1000):
    dname = '/home/malyatha'
    train_paths = [("{dir}/train_npy/{num}.npy".format(dir=dname, num=num),
                    "{dir}/train_gt_npy/{num}_gt.npy".format(dir=dname, num=num))
                    for num in range(1, 18)]
    valid_paths = [("{dir}/test_npy/{num}.npy".format(dir=dname, num=num),
                    "{dir}/test_gt_npy/{num}_gt.npy".format(dir=dname, num=num))
                    for num in range(1, 18)]

    blk_train = DataBlks(train_paths, d, h, w, batchsize, min_density=min_density, num_patch_per_img=num_patch_per_img, rotate=True)
    blk_valid = DataBlks(valid_paths, d, h, w, batchsize, min_density=min_density, num_patch_per_img=num_patch_per_img, rotate=False)
    return blk_train.make_data(), blk_valid.make_data()






if __name__ == '__main__':
    # data = Iris()
    # data = Skin(train_valid=[5,1], shuffle=True, batchsize=32)
    # data.make_data()
    # import pdb; pdb.set_trace()
    print(list(rotations12(np.random.rand(10,10,10))))
