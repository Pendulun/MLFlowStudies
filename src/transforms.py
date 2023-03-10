from torchvision import transforms

def get_trivial_augment(magnitude:int = 31):
    """
    Return TrivialAugmentWide
    """
    return transforms.TrivialAugmentWide(magnitude)

def get_rand_augment(num_ops:int=2, magnitude:int=9):
    """
    Return RandAugment
    """
    return transforms.RandAugment(num_ops=num_ops, magnitude=magnitude)

def get_custom_augment(random_rotation_params={'degrees':[0, -15, 15, -30, 30, -45, 45, -60, 60]},
                       color_jitter_params={'brightness':0.5, 'contrast':0.5, 'hue':0.5},
                       gaussian_blur_params={'kernel_size':(5, 9)},
                       random_horizontal_flip_params={'p':0.5},
                       random_perspective_params={}):
    """
    Returns a custom transform composed of:
    RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomPerspective, GaussianBlur
    """
    custom_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(**random_horizontal_flip_params),
        transforms.RandomRotation(**random_rotation_params),
        transforms.ColorJitter(**color_jitter_params),
        transforms.RandomPerspective(**random_perspective_params),
        transforms.GaussianBlur(**gaussian_blur_params)
    ])
    return custom_transform