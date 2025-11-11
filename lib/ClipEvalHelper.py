import clip

from PIL import Image 

from torchvision.transforms.functional import adjust_contrast

from lib.ClipHelper import *
from lib.template import imagenet_templates
class ClipEval():
    def __init__(self):
        
        device='cuda'
        self.clip_model, self.preprocess = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.requires_grad_(False)
        


        ## not inited yet
        # self.source_features = None
    
    def getImageLatent(self,content_image_ori):
        source_features = self.clip_model.encode_image(clip_normalize(content_image_ori,'cuda'))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        return source_features 

    def getCosSim_fromLatent(self,latent1,latent2):
        return torch.cosine_similarity(latent1, latent2, dim=1)


    def getCosSim(self,image1,image2):
        return self.getCosSim_fromLatent(self.getImageLatent(image1), self.getImageLatent(image2)) 
