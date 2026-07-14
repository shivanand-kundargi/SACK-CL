import os
import importlib.util
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from typing import Tuple
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from utils import smart_joint
from utils.conf import base_path
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode


def _cub_npz_ready(root: str) -> bool:
    return os.path.isfile(os.path.join(root, 'train_data.npz')) and os.path.isfile(os.path.join(root, 'test_data.npz'))


def _prepare_cub_npz(root: str, force: bool = False) -> None:
    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_cub200_npz.py'
    spec = importlib.util.spec_from_file_location("prepare_cub200_npz", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load CUB preparation script at {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(root=Path(root), force=force)


def _load_cub_npz(root: str, train: bool, download: bool):
    split_file = 'train_data.npz' if train else 'test_data.npz'
    npz_path = smart_joint(root, split_file)
    try:
        return np.load(npz_path, allow_pickle=True)
    except ModuleNotFoundError as exc:
        if exc.name != 'numpy._core' or not download:
            raise
        print('CUB-200 NPZ files were written by an incompatible NumPy version; rebuilding with this Python.')
        _prepare_cub_npz(root, force=True)
        return np.load(npz_path, allow_pickle=True)


def _load_cub_arrays(root: str, train: bool, download: bool):
    data_file = _load_cub_npz(root, train=train, download=download)
    data = data_file['data']
    if getattr(data, 'ndim', None) != 4:
        if not download:
            raise RuntimeError(
                f"CUB-200 data in {root} has shape {getattr(data, 'shape', None)}; "
                "expected a fixed-size 4D uint8 array."
            )
        print('CUB-200 NPZ files use the old object-array format; rebuilding fixed-size arrays for replay.')
        _prepare_cub_npz(root, force=True)
        data_file = _load_cub_npz(root, train=train, download=False)
        data = data_file['data']
        if getattr(data, 'ndim', None) != 4:
            raise RuntimeError(
                f"CUB-200 rebuild produced shape {getattr(data, 'shape', None)}; expected a 4D uint8 array."
            )
    return data_file, data


class MyCUB200(Dataset):
    """
    Overrides dataset to change the getitem function.
    """
    IMG_SIZE = 224  # Default, will be updated in get_data_loaders
    N_CLASSES = 200
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download and not _cub_npz_ready(root):
            print('Preparing CUB-200-2011 NPZ files from the official CaltechDATA archive.')
            try:
                _prepare_cub_npz(root)
            except Exception as exc:
                raise RuntimeError(
                    "CUB-200-2011 data is missing and automatic preparation failed. "
                    f"Expected {smart_joint(root, 'train_data.npz')} and {smart_joint(root, 'test_data.npz')}. "
                    "You can prepare them manually with: "
                    f"python scripts/prepare_cub200_npz.py --root {root}. "
                    "If the compute node has no internet, download CUB_200_2011.tgz from "
                    "https://data.caltech.edu/records/65de6-vp158 and run the same command with "
                    "--archive /path/to/CUB_200_2011.tgz."
                ) from exc

        if not _cub_npz_ready(root):
            raise FileNotFoundError(
                f"CUB-200-2011 NPZ files not found in {root}. Run "
                f"`python scripts/prepare_cub200_npz.py --root {root}` first."
            )

        data_file, data = _load_cub_arrays(root, train=train, download=download)

        self.data = data
        self.targets = torch.from_numpy(data_file['targets']).long()
        self.classes = data_file['classes']
        self.segs = data_file['segs']
        self._return_segmask = False

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, 'logits') else [
            img, target, not_aug_img]

        if self._return_segmask:
            # TODO: add to the return tuple
            raise "Unsupported segmentation output in training set!"

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)


class CUB200(MyCUB200):
    """Base CUB200 dataset."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.

        Args:
            index: index of the element to be returned

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, 'logits') else [img, target]

        if ret_segmask or self._return_segmask:
            # TODO: does not work with the current implementation
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode='L')
            seg = transforms.ToTensor()(transforms.CenterCrop((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple


class SequentialCUB200(ContinualDataset):
    """Sequential CUB200 Dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformation to apply to the data.
        TEST_TRANSFORM (torchvision.transforms): transformation to apply to the test data.
    """
    NAME = 'seq-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    TRANSFORM = transforms.Compose([
        transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)])
    TEST_TRANSFORM = transforms.Compose([transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                         transforms.CenterCrop(MyCUB200.IMG_SIZE),
                                         transforms.ToTensor(),
                                         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        # Set IMG_SIZE based on model before creating datasets
        # self.args is available because ContinualDataset.__init__ sets it
        if hasattr(self, 'args') and hasattr(self.args, 'model') and self.args.model:
            if self.args.model.lower().startswith('icarl'):
                MyCUB200.IMG_SIZE = 256
            else:
                MyCUB200.IMG_SIZE = 224
            
            # Update SIZE and recreate TRANSFORM/TEST_TRANSFORM with correct size
            SequentialCUB200.SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)
            SequentialCUB200.TRANSFORM = transforms.Compose([
                transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(SequentialCUB200.SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)])
            SequentialCUB200.TEST_TRANSFORM = transforms.Compose([
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(MyCUB200.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)])
        
        train_dataset = MyCUB200(base_path() + 'CUB200', train=True,
                                 download=True, transform=SequentialCUB200.TRANSFORM)
        test_dataset = CUB200(base_path() + 'CUB200', train=False,
                              download=True, transform=SequentialCUB200.TEST_TRANSFORM)

        train, test = store_masked_loaders(
            train_dataset, test_dataset, self)

        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCUB200.TRANSFORM])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200.MEAN, SequentialCUB200.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCUB200.MEAN, SequentialCUB200.STD)
        return transform

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 128

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names


CLASS_NAMES = [
    'black footed albatross',
    'laysan albatross',
    'sooty albatross',
    'groove billed ani',
    'crested auklet',
    'least auklet',
    'parakeet auklet',
    'rhinoceros auklet',
    'brewer blackbird',
    'red winged blackbird',
    'rusty blackbird',
    'yellow headed blackbird',
    'bobolink',
    'indigo bunting',
    'lazuli bunting',
    'painted bunting',
    'cardinal',
    'spotted catbird',
    'gray catbird',
    'yellow breasted chat',
    'eastern towhee',
    'chuck will widow',
    'brandt cormorant',
    'red faced cormorant',
    'pelagic cormorant',
    'bronzed cowbird',
    'shiny cowbird',
    'brown creeper',
    'american crow',
    'fish crow',
    'black billed cuckoo',
    'mangrove cuckoo',
    'yellow billed cuckoo',
    'gray crowned rosy finch',
    'purple finch',
    'northern flicker',
    'acadian flycatcher',
    'great crested flycatcher',
    'least flycatcher',
    'olive sided flycatcher',
    'scissor tailed flycatcher',
    'vermilion flycatcher',
    'yellow bellied flycatcher',
    'frigatebird',
    'northern fulmar',
    'gadwall',
    'american goldfinch',
    'european goldfinch',
    'boat tailed grackle',
    'eared grebe',
    'horned grebe',
    'pied billed grebe',
    'western grebe',
    'blue grosbeak',
    'evening grosbeak',
    'pine grosbeak',
    'rose breasted grosbeak',
    'pigeon guillemot',
    'california gull',
    'glaucous winged gull',
    'heermann gull',
    'herring gull',
    'ivory gull',
    'ring billed gull',
    'slaty backed gull',
    'western gull',
    'anna hummingbird',
    'ruby throated hummingbird',
    'rufous hummingbird',
    'green violetear',
    'long tailed jaeger',
    'pomarine jaeger',
    'blue jay',
    'florida jay',
    'green jay',
    'dark eyed junco',
    'tropical kingbird',
    'gray kingbird',
    'belted kingfisher',
    'green kingfisher',
    'pied kingfisher',
    'ringed kingfisher',
    'white breasted kingfisher',
    'red legged kittiwake',
    'horned lark',
    'pacific loon',
    'mallard',
    'western meadowlark',
    'hooded merganser',
    'red breasted merganser',
    'mockingbird',
    'nighthawk',
    'clark nutcracker',
    'white breasted nuthatch',
    'baltimore oriole',
    'hooded oriole',
    'orchard oriole',
    'scott oriole',
    'ovenbird',
    'brown pelican',
    'white pelican',
    'western wood pewee',
    'sayornis',
    'american pipit',
    'whip poor will',
    'horned puffin',
    'common raven',
    'white necked raven',
    'american redstart',
    'geococcyx',
    'loggerhead shrike',
    'great grey shrike',
    'baird sparrow',
    'black throated sparrow',
    'brewer sparrow',
    'chipping sparrow',
    'clay colored sparrow',
    'house sparrow',
    'field sparrow',
    'fox sparrow',
    'grasshopper sparrow',
    'harris sparrow',
    'henslow sparrow',
    'le conte sparrow',
    'lincoln sparrow',
    'nelson sharp tailed sparrow',
    'savannah sparrow',
    'seaside sparrow',
    'song sparrow',
    'tree sparrow',
    'vesper sparrow',
    'white crowned sparrow',
    'white throated sparrow',
    'cape glossy starling',
    'bank swallow',
    'barn swallow',
    'cliff swallow',
    'tree swallow',
    'scarlet tanager',
    'summer tanager',
    'artic tern',
    'black tern',
    'caspian tern',
    'common tern',
    'elegant tern',
    'forsters tern',
    'least tern',
    'green tailed towhee',
    'brown thrasher',
    'sage thrasher',
    'black capped vireo',
    'blue headed vireo',
    'philadelphia vireo',
    'red eyed vireo',
    'warbling vireo',
    'white eyed vireo',
    'yellow throated vireo',
    'bay breasted warbler',
    'black and white warbler',
    'black throated blue warbler',
    'blue winged warbler',
    'canada warbler',
    'cape may warbler',
    'cerulean warbler',
    'chestnut sided warbler',
    'golden winged warbler',
    'hooded warbler',
    'kentucky warbler',
    'magnolia warbler',
    'mourning warbler',
    'myrtle warbler',
    'nashville warbler',
    'orange crowned warbler',
    'palm warbler',
    'pine warbler',
    'prairie warbler',
    'prothonotary warbler',
    'swainson warbler',
    'tennessee warbler',
    'wilson warbler',
    'worm eating warbler',
    'yellow warbler',
    'northern waterthrush',
    'louisiana waterthrush',
    'bohemian waxwing',
    'cedar waxwing',
    'american three toed woodpecker',
    'pileated woodpecker',
    'red bellied woodpecker',
    'red cockaded woodpecker',
    'red headed woodpecker',
    'downy woodpecker',
    'bewick wren',
    'cactus wren',
    'carolina wren',
    'house wren',
    'marsh wren',
    'rock wren',
    'winter wren',
    'common yellowthroat'
]
