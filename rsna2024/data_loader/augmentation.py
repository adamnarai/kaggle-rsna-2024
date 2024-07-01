import albumentations as A
    
class Normalize():
    def __init__(self):
        self.aug = A.Compose([A.Normalize(mean=0.5, std=0.5)])
    
    def get_transform(self):
        return self.aug

class CoarseDropout():
    def __init__(self, p=0.5):
        coarse_dropout = A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=p)
        self.aug = A.Compose([coarse_dropout, A.Normalize(mean=0.5, std=0.5)])
        
    def get_transform(self):
        return self.aug
