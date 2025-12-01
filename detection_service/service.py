import bentoml
import numpy as np
import torch
from PIL import Image as PILImage
from typing import List

from preprocess import preprocess_images
from postprocess import non_max_suppression_face, rescale_detections
from model_loader import create_session


@bentoml.service()
class FaceDetectionBatchService:
    def __init__(self):
        self.session = create_session("yolov6s_face.onnx")
        self.input_name = self.session.get_inputs()[0].name

    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=300)
    async def detect_batch(self, images: List[PILImage.Image]) -> List[List[dict]]:
        imgs_bgr = [np.array(img)[:, :, ::-1] for img in images]

        _, prepared_data = preprocess_images(imgs_bgr)

        batch_tensor = np.stack([data[0][0] for data in prepared_data])
        batch_tensor = batch_tensor.astype(np.float32) / 255.0

        ort_outs = self.session.run(None, {self.input_name: batch_tensor})
        predictions = torch.from_numpy(ort_outs[0])

        results = []
        for i, pred in enumerate(predictions):
            dets = non_max_suppression_face(
                pred.unsqueeze(0),
                conf_thres=0.4,
                iou_thres=0.45,
                max_det=300
            )
            final = rescale_detections(dets, [prepared_data[i]])
            results.append(final[0])

        return results
    
@bentoml.service()
class FaceDetectionService:
    batch_service = bentoml.depends(FaceDetectionBatchService)

    @bentoml.api()
    async def detect(self, image: PILImage.Image) -> List[dict]:
        batch_result = await self.batch_service.to_async.detect_batch([image])
        return batch_result[0]