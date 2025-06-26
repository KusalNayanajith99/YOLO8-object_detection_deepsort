import torch
import torchreid
from torchvision import transforms

class OSNetExtractor:
    def __init__(self, model_name='osnet_x1_0',	device='cuda'):
        self.device = device
        # Load the pre-trained model
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # Placeholder, not used for feature extraction
            loss='softmax',
            pretrained=True
        )
        self.model.to(self.device)
        self.model.eval()

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    @torch.no_grad()
    def extract_features(self, image_crops):
        """
        Extracts features from a list of image crops.
        Args:
            image_crops (list): A list of person image crops (numpy arrays).
        Returns:
            torch.Tensor: A tensor of feature vectors.
        """
        if not image_crops:
            return torch.empty((0, 512)).to(self.device)
        
        # Prepare a batch of images
        image_batch = torch.stack(
            [self.transform(crop) for crop in image_crops]
        ).to(self.device)

        # Extract features
        features = self.model(image_batch)
        return features.cpu() # Return features on CPU
