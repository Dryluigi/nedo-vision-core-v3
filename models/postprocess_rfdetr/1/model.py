import triton_python_backend_utils as pb_utils
import torch
import torchvision
import torch.utils.dlpack
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args['model_config'])
        self.max_output_boxes = 100 
        self.iou_threshold = 0.45
        self.score_threshold = 0.25

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get input tensor (BATCH, 8400, 6)
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            
            # Zero-copy conversion to Torch via DLPack
            # If input is on GPU, this stays on GPU
            input_data = torch.from_dlpack(in_tensor.to_dlpack())
            
            batch_size = input_data.shape[0]
            
            # Prepare output tensors
            # We must ensure we create tensors on the same device as input
            device = input_data.device
            
            out_num = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
            out_boxes = torch.zeros((batch_size, self.max_output_boxes, 4), dtype=torch.float32, device=device)
            out_scores = torch.zeros((batch_size, self.max_output_boxes), dtype=torch.float32, device=device)
            out_classes = torch.zeros((batch_size, self.max_output_boxes), dtype=torch.float32, device=device)

            for i in range(batch_size):
                prediction = input_data[i] # [8400, 6]
                
                # Filter by score
                mask = prediction[:, 4] > self.score_threshold
                pred_filtered = prediction[mask]
                
                if pred_filtered.shape[0] == 0:
                    continue
                
                boxes = pred_filtered[:, :4]
                scores = pred_filtered[:, 4]
                classes = pred_filtered[:, 5]
                
                # NMS (GPU accelerated if input is on GPU)
                keep_indices = torchvision.ops.nms(boxes, scores, self.iou_threshold)
                
                # Limit to max_output_boxes
                if keep_indices.shape[0] > self.max_output_boxes:
                    keep_indices = keep_indices[:self.max_output_boxes]
                
                final_boxes = boxes[keep_indices]
                final_scores = scores[keep_indices]
                final_classes = classes[keep_indices]
                num_dets = keep_indices.shape[0]
                
                out_num[i, 0] = num_dets
                out_boxes[i, :num_dets] = final_boxes
                out_scores[i, :num_dets] = final_scores
                out_classes[i, :num_dets] = final_classes

            # Convert back to Triton Tensors via DLPack (Zero Copy)
            out_tensor_num = pb_utils.Tensor.from_dlpack("num_detections", torch.utils.dlpack.to_dlpack(out_num))
            out_tensor_boxes = pb_utils.Tensor.from_dlpack("detection_boxes", torch.utils.dlpack.to_dlpack(out_boxes))
            out_tensor_scores = pb_utils.Tensor.from_dlpack("detection_scores", torch.utils.dlpack.to_dlpack(out_scores))
            out_tensor_classes = pb_utils.Tensor.from_dlpack("detection_classes", torch.utils.dlpack.to_dlpack(out_classes))

            responses.append(pb_utils.InferenceResponse([
                out_tensor_num, out_tensor_boxes, out_tensor_scores, out_tensor_classes
            ]))

        return responses

    def finalize(self):
        pass
