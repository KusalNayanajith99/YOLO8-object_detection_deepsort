import torch
import torchvision.transforms as T
import numpy as np
 
from torchreid.models.osnet import osnet_x1_0            # or your import
from torchreid.utils import load_pretrained_weights      # if needed

class OSNetExtractor:
    def __init__(self, model_name='osnet_x1_0', device='cpu'):
        self.device = device
        self.model  = osnet_x1_0(num_classes=1, pretrained=True)  # or your loading code
        self.model.eval().to(device)

        # same transforms YOLO used when the model was trained
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    # ---------- NEW ----------
    @torch.no_grad()
    def extract(self, frame_bgr, bboxes):
        """
        Parameters
        ----------
        frame_bgr : np.ndarray  (H,W,3)  OpenCV BGR frame
        bboxes     : list[[x1,y1,x2,y2], ...]  coordinates in the same frame

        Returns
        -------
        features : list[np.ndarray]  L2–normalised 512-D OSNet embeddings
        """
        if not bboxes:
            return []

        patches = []
        for (x1, y1, x2, y2) in bboxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:                          # safety check
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            patches.append(self.transforms(crop) )

        batch = torch.stack(patches).to(self.device)
        feats = self.model(batch)                       # (N, 512)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().tolist()
