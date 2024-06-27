import albumentations as A

class Normalize():
    def __init__(self):
        self.aug = A.Compose([A.Normalize(mean=0.5, std=0.5)])

    def __call__(self, image):
        return self.aug(image=image)['image']