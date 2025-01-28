import numpy as np
from segment_anything import build_sam, SamPredictor


class SAM:
    """This is a bounding box conditioned open segmentor that segments the main object in the given bbox"""
    def __init__(self, device):
        sam_checkpoint = '/mnt/storage_ssd/big_files/sam_vit_h_4b8939.pth'
        self.sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
        self.device = device

    @staticmethod
    def draw_mask(mask, image, random_color=True):
        if random_color:
            color = np.concatenate((np.random.random(3), np.array([0.8])), axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        from PIL import Image
        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def predict(self, image, boxe):
        self.sam_predictor.set_image(image)
        H, W, _ = image.shape
        # boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxe.to(self.device), image.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        masks = masks[0][0].cpu().numpy().astype(np.uint8)
        return masks, SAM.draw_mask(masks, image)
