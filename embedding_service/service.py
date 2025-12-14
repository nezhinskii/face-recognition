import numpy as np
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any, List
import bentoml
import cv2

from preprocess import preprocess_image, extract_largest_face_aligned
from model_loader import create_session

class BatchInput(BaseModel):
    image: Path
    detections: List[Dict[str, Any]]

@bentoml.service()
class FaceEmbeddingBatchService:

    def __init__(self):
        self.session = create_session("LVFace-S_Glint360K.onnx")
        self.input_name = self.session.get_inputs()[0].name

    @bentoml.api(batchable=True, max_batch_size=16, max_latency_ms=300)
    async def embed_batch(self, inputs: List[BatchInput]) -> List[Dict[str, Any]]:
        img_bgr_list = [cv2.imread(str(inp.image)) for inp in inputs]
        detections_list = [inp.detections for inp in inputs]

        aligned_faces: List[np.ndarray | None] = []
        best_det_ids: List[int] = []
        for img_bgr, dets in zip(img_bgr_list, detections_list):
            face, best_det_id = extract_largest_face_aligned(img_bgr, dets, output_size=112)
            aligned_faces.append(face)
            best_det_ids.append(int(best_det_id))

        batch_tensors = []
        valid_indices = []
        for idx, face in enumerate(aligned_faces):
            if face is not None:
                tensor = preprocess_image(face)
                flipped = cv2.flip(face, 1)
                tensor_flip = preprocess_image(flipped)
                batch_tensors.append(tensor)
                batch_tensors.append(tensor_flip)
                valid_indices.append(idx)

        results = [None] * len(inputs)
        if batch_tensors:
            batch_input = np.vstack(batch_tensors)

            embeddings: np.ndarray = self.session.run(None, {self.input_name: batch_input})[0]

            num_valid = len(valid_indices)
            for i in range(num_valid):
                orig_idx = valid_indices[i]
                emb_orig = embeddings[2 * i]
                emb_flip = embeddings[2 * i + 1]
                avg_emb = (emb_orig + emb_flip) / 2.0
                norm_avg = np.linalg.norm(avg_emb)
                final_emb = avg_emb / norm_avg if norm_avg != 0 else avg_emb
                results[orig_idx] = {
                    "embedding": final_emb.tolist(),
                    "best_det_id": best_det_ids[i]
                }

        for i, res in enumerate(results):
            if res is None:
                results[i] = {"error": "no_face"}
        
        return results

@bentoml.service()
class FaceEmbeddingService:
    batch_service = bentoml.depends(FaceEmbeddingBatchService)

    @bentoml.api()
    async def embed(self, image: Path, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_input = [BatchInput(image=image, detections=detections)]
        results = await self.batch_service.to_async.embed_batch(batch_input)
        return results[0]