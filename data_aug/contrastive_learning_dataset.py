from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, WatermarkViewGenerator
import os


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.ToTensor()])
        return data_transforms


    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                          split='train',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(
                                                                  32),
                                                              n_views),
                                                          download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet256/",
                              split='train',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_simclr_pipeline_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), 
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet256/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_simclr_pipeline_transform(
                                      224),
                                  n_views))
                          }


        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
        
        
class RegularDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        data_transforms = transforms.Compose([transforms.ToTensor()])
        return data_transforms

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True, 
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32), n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='train',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='train',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()
    def get_test_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=False,
                                                              transform=ContrastiveLearningViewGenerator(  
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='test',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(32),
                                                              n_views),
                                                          download=True),

                          'svhn': lambda: datasets.SVHN(self.root_folder+"/SVHN",
                                                        split='test',
                                                        transform=ContrastiveLearningViewGenerator(
                                                            self.get_simclr_pipeline_transform(
                                                                32),
                                                            n_views),
                                                        download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='val',
                              transform=ContrastiveLearningViewGenerator(
                                  self.get_imagenet_transform(
                                      224),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()


class WatermarkDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_transform():
        data_transform1 = transforms.Compose([transforms.RandomRotation(degrees=(0, 180)),
                                              transforms.ToTensor()])
        data_transform2 = transforms.Compose([transforms.RandomRotation(degrees=(180, 360)),
                                              transforms.ToTensor()])
        return [data_transform1, data_transform2]

    @staticmethod
    def get_imagenet_transform(size, s=1):
        data_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.ToTensor()])
        data_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomRotation(degrees=(180, 360)),
             transforms.ToTensor()])
        return [data_transform1, data_transform2]

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=WatermarkViewGenerator(
                                                                  self.get_transform(),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(f"/checkpoint/{os.getenv('USER')}/SimCLR/stl10", split='unlabeled',
                                                          transform=WatermarkViewGenerator(
                                                              self.get_transform(),
                                                              n_views),
                                                          download=True),
                          'svhn': lambda: datasets.SVHN(
                              self.root_folder + "/SVHN",
                              split='test',
                              transform=WatermarkViewGenerator(
                                  self.get_transform(),
                                  n_views),
                              download=True),
                          'imagenet': lambda: datasets.ImageNet(
                              root="/scratch/ssd002/datasets/imagenet_pytorch/",
                              split='val',
                              transform=WatermarkViewGenerator(
                                  self.get_imagenet_transform(
                                      32),
                                  n_views))
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise Exception()
        else:
            return dataset_fn()