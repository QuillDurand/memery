__all__ = ['make_dataset', 'pil_loader', 'DatasetImagePaths', 'clip_transform', 'crafter', 'preproc']


import torch
from torchvision.datasets import VisionDataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def make_dataset(new_files):
    '''Returns a list of samples of a form (path_to_sample, class) and in
    this case the class is just the filename'''
    samples = []
    slugs = []
    for i, f in enumerate(new_files):
        path, slug = f
        samples.append((str(path), i))
        slugs.append((slug, i))
    return(samples, slugs)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetImagePaths(VisionDataset):
    def __init__(self, new_files, transforms = None):
        super(DatasetImagePaths, self).__init__(new_files, transforms=transforms)
        samples, slugs = make_dataset(new_files)
        self.samples = samples
        self.slugs = slugs
        self.loader = pil_loader
        self.root = 'file dataset'
    def __len__(self):
        return(len(self.samples))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is not None:
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample, target


from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def crafter(new_files, device, batch_size=128, num_workers=4):
    with torch.no_grad():
        imagefiles=DatasetImagePaths(new_files, clip_transform(224))
        img_loader=torch.utils.data.DataLoader(imagefiles, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return(img_loader)


def preproc(img):
    transformed = clip_transform(224)(img)
    return(transformed)