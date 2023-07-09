import sys
from models.matching import Matching
from models.utils import frame2tensor
import cv2
import numpy as np
import torch


class SuperGlueWrapper:
    
    def __init__(
        self,
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        weights="indoor",
        sinkhorn_iterations=20,
        match_threshold=0.2,
        device="cuda",
    ):
        config = {
            "superpoint": {
                "nms_radius": nms_radius,
                "keypoint_threshold": keypoint_threshold,
                "max_keypoints": max_keypoints,
            },
            "superglue": {
                "weights": weights,
                "sinkhorn_iterations": sinkhorn_iterations,
                "match_threshold": match_threshold,
            },
        }

        self.keys = ['keypoints', 'scores', 'descriptors']
        self.device = device
        self.model = Matching(config).eval().to(device)
        self.last_keypoints = None

    def set_config(
        self,
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=-1,
        sinkhorn_iterations=20,
        match_threshold=0.2
    ):
        self.model.superpoint.config['nms_radius'] = nms_radius
        self.model.superpoint.config['keypoint_threshold'] = keypoint_threshold
        self.model.superpoint.config['max_keypoints'] = max_keypoints
        self.model.superglue.config['sinkhorn_iterations'] = sinkhorn_iterations
        self.model.superglue.config['match_threshold'] = match_threshold
    
    def get_init_keypoints(self, img: cv2.Mat):
        tensor = frame2tensor(img, self.device)
        self.last_keypoints = self.model.superpoint({'image': tensor})
        self.last_keypoints = {k+'0': self.last_keypoints[k] for k in self.keys}
        self.last_keypoints['image0'] = tensor

    def forward_full(self, img_0: cv2.Mat, img_1: cv2.Mat):
        tensor_0 = frame2tensor(img_0, self.device)
        tensor_1 = frame2tensor(img_1, self.device)
        pred = self.model({"image0": tensor_0, "image1": tensor_1})
        kpts0 = pred["keypoints0"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].detach().cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        return kpts0, matches, confidence, kpts1
        
    def forward_append(self, img: cv2.Mat):
        tensor = frame2tensor(img, self.device)
        pred = self.model({**self.last_keypoints, "image1": tensor})
        kpts0 = self.last_keypoints["keypoints0"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].detach().cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()

        self.last_keypoints = {k+'0': pred[k+'1'] for k in self.keys}
        self.last_keypoints['image0'] = tensor
        return kpts0, matches, confidence, kpts1

    def get_keypoints(self, img: cv2.Mat):
        x = frame2tensor(img, self.device)
        y = self.model.superpoint({'image': x})
        kpts = y["keypoints"][0].cpu().numpy()
        scores = y["scores"][0].detach().cpu().numpy().reshape(1, -1)
        des = y["descriptors"][0].detach().cpu().numpy()
        frame_tensor = x.cpu().numpy()
        return kpts, scores, des, frame_tensor

    def match(
            self,
            kpts0: np.array, scores0: np.array, des0: np.array, im_tensor0: np.array,
            kpts1: np.array, scores1: np.array, des1: np.array, im_tensor1: np.array
        ):
        keypoints0 = torch.from_numpy(kpts0).to(self.device)
        scores0 = torch.from_numpy(scores0).to(self.device)
        descriptors0 = torch.from_numpy(des0).to(self.device)
        image0 = torch.from_numpy(im_tensor0).to(self.device)

        keypoints1 = torch.from_numpy(kpts1).to(self.device)
        scores1 = torch.from_numpy(scores1).to(self.device)
        descriptors1 = torch.from_numpy(des1).to(self.device)
        image1 = torch.from_numpy(im_tensor1).to(self.device)
        
        data = {
            "keypoints0": keypoints0,
            "scores0": scores0,
            "descriptors0": descriptors0,
            "image0": image0,
            "keypoints1": keypoints1,
            "scores1": scores1,
            "descriptors1": descriptors1,
            "image1": image1
        }
        pred = self.model(data)
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].detach().cpu().numpy()
        return kpts0, matches, confidence, kpts1
    