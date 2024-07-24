import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class Indoor(Dataset):
    """`Indoor Scene Recognition <https://web.mit.edu/torralba/www/indoor.html>`_ Dataset.
        Args:
            root (string|Path): Root directory of dataset
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            download (bool, optional): If true, downloads the dataset tar files from the internet and
                puts it in root directory. If the tar files are already downloaded, they are not
                downloaded again.
        """
    folder = 'Indoor'

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):

        self.root = Path(root) #join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.classes = sorted([x.name for x in (self.root / "Images").iterdir()])

        split = self.load_split()

        self.images_folder = self.root / 'Images'

        self.data = [(x[0], self.classes.index(x[1])) for x in split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self.data[index]
        target_class = int(target_class)
        image_path = (self.images_folder / image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if (self.root / 'Images').exists():
            if len(list((self.root / 'Images').iterdir())) == 67:
                print('Files already downloaded and verified')
                return
        if not self.root.exists():
            self.root.mkdir()

        im_url = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
        tar_filename = "indoorCVPR_09.tar"
        download_url(im_url, self.root, tar_filename, None)
        print('Extracting downloaded file: ' + str(self.root / tar_filename))
        with tarfile.open((self.root / tar_filename), 'r') as tar_file:
            tar_file.extractall(self.root)
        os.remove((self.root / tar_filename))

        print("downloading train/test lists")
        download_url("https://web.mit.edu/torralba/www/TrainImages.txt", self.root, "TrainImages.txt", None)
        download_url("https://web.mit.edu/torralba/www/TestImages.txt", self.root, "TestImages.txt", None)
        print("downloading complete.")

    def load_split(self):
        with open(self.root / "TestImages.txt") as file:
            filelist = file.readlines()
        filelist = [x.strip() for x in filelist]

        if self.train:
            # the train filelist doesn't contain all possible train images, so we deduce them from the test set
            all_images = []
            for folder in (self.root / "Images").iterdir():
                all_images.extend(folder.iterdir())
            test_im_set = set([tuple(x.split("/")) for x in filelist])
            all_im_set = set([x.parts[-2:] for x in all_images])
            train_im_set = all_im_set - test_im_set
            filelist = ["/".join(x) for x in train_im_set]

        labels = [x.split("/")[0] for x in filelist]
        return list(zip(filelist, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.data)):
            image_name, target_class = self.data[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self.data), len(counts.keys()),
                                                                     float(len(self.data)) / float(
                                                                         len(counts.keys()))))

        return counts


def load_indoor(config, augment=False):
    from binary_nn.datasets.commons import get_train_transforms, get_test_transforms
    indoor_path = Path(config["indoor"]["path"]).absolute()
    train_transforms = get_train_transforms(augment)
    test_transforms = get_test_transforms()

    ds = Indoor(indoor_path, train=True, transform=train_transforms, download=True)
    test_ds = Indoor(indoor_path, train=False, transform=test_transforms)
    ds.stats()
    n_classes = 67
    return n_classes, ds, test_ds


if __name__ == '__main__':
    ds = Indoor("C:\\data\\datasets\\Indoor", download=True)
    # ds = Indoor("C:\\data\\datasets\\Indoor", download=True)
    ds.stats()
    im = ds[0]
    pass
