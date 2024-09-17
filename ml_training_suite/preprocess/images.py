from ml_training_suite.preprocess import PreProcess, Select
from torch import nn
import numpy as np

class RescaleColor(nn.Module):
    def __init__(
            self,
            minimum=0,
            maximum=255,
            *args,
            **kwargs) -> None:
        """
        Rescale inputs of a preset range to a range from 0 to 1.

        Args:
            minimum: the smallest possible value of expected inputs.
            maximum: the largest possible value of expected inputs.
        """
        super().__init__(*args, **kwargs)
        self.min = minimum
        self.max = maximum
    
    def forward(self, x):
        return (x - self.min) / (self.max - self.min)

class Pad(nn.Module):
    def __init__(
            self,
            width=300,
            height=300,
            mode='edge',
            pass_larger_images:bool = False,
            *args,
            **kwargs) -> None:
        """
        Add pixels to images to provide a consistent size.

        Args:
            width: The desired total pixel width.
            height: The desired total pixel height.
            mode: The means by which the image should be padded.
        """
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.mode = mode
        self.pass_larger_images = pass_larger_images
    
    def forward(self, x):
        h,w,c = x.shape
        pad_h = self.height-h
        pad_w = self.width-w
        if self.pass_larger_images:
            pad_h = max(pad_h,0)
            pad_w = max(pad_w,0)
        return np.pad(
            x,
            ((0,pad_h),(0,pad_w),(0,0)),
            mode=self.mode)
    
class Crop(nn.Module):
    def __init__(
            self,
            width=125,
            height=125,
            seed=None,
            *args,
            **kwargs) -> None:
        """
        Take away pixels from images to provide a consistent size.

        Args:
            width: The desired total pixel width.
            height: The desired total pixel height.
            seed: Seed for rng to create reproducible results if desired.
        """
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed=seed)
    
    def forward(self, x):
        h,w,c = x.shape
        start_x = self.rng.integers(w-self.width)
        start_y = self.rng.integers(h-self.height)
        return x[start_x:start_x+self.width, start_y:start_y+self.height, :]

class RandomFlip(nn.Module):
    def __init__(
            self,
            seed=None,
            *args,
            **kwargs) -> None:
        """
        Take away pixels from images to provide a consistent size.

        Args:
            width: The desired total pixel width.
            height: The desired total pixel height.
            seed: Seed for rng to create reproducible results if desired.
        """
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
    
    def forward(self, x):
        if self.rng.integers(2):
            x = np.flip(x, -2)
        if self.rng.integers(2):
            x = np.flip(x, -3)
        return x

class RandomBrightness(nn.Module):
    def __init__(
            self,
            max_inc=50,
            max_dec=50,
            seed=None,
            *args,
            **kwargs) -> None:
        """
        Add to image colors for greater or lesser brightness, randomly.

        Args:
            max_inc: The highest brightness addition to apply.
            max_dec: The lowest brightness addition to apply.
            seed: Seed for rng to create reproducible results if desired.
        """
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
        self.max_inc = max_inc
        self.max_dec = max_dec
    
    def forward(self, x):
        change = self.rng.integers(self.max_dec, self.max_inc, endpoint=True)
        x = x+change
        x[x<0] = 0
        x[x>255] = 255
        return x

class RandomContrast(nn.Module):
    def __init__(
            self,
            max_inc=2,
            max_dec=0.5,
            seed=None,
            *args,
            **kwargs) -> None:
        """
        Scale image colors for greater or lesser contrast, randomly.

        Args:
            max_inc: The highest contrastive scalar to apply.
            max_dec: The lowest contrastive scalar to apply.
            seed: Seed for rng to create reproducible results if desired.
        """
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
        self.max_inc = max_inc
        self.max_dec = max_dec
    
    def forward(self, x):
        change = self.rng.random()
        change = (self.max_inc-self.max_dec)*change+self.max_dec
        x = x*change
        x[x<0] = 0
        x[x>255] = 255
        return x

class PPPicture(PreProcess):
    def __init__(
        self,
        random_brightness:bool = False,
        random_contrast:bool = False,
        rescale_images:bool = True,
        random_flips:bool = False,
        omit:bool = False,
        pad_mode:str = None,
        pad_width=200,
        pad_height=200,
        pass_larger_images:bool = False,
        crop:bool = False,
        crop_width=125,
        crop_height=125,
        seed:int = None,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rand_brit = RandomBrightness(seed=seed) if random_brightness else None
        self.rand_cont = RandomContrast(seed=seed) if random_contrast else None
        self.rescale = RescaleColor() if rescale_images else None
        self.select = Select(do_not_include=omit) if omit else None
        self.rand_flip = RandomFlip(seed=seed) if random_flips else None
        self.pad = Pad(
            width=pad_width,
            height=pad_height,
            mode=pad_mode,
            pass_larger_images=pass_larger_images) if pad_mode else None
        self.crop = Crop(width=crop_width, height=crop_height, seed=seed) if crop else None