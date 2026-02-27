#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "nvdsinfer_custom_impl.h"

extern "C"
bool NvDsInferParseCustomTritonYolo(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    const float *num_detections = nullptr;
    const float *detection_boxes = nullptr;
    const float *detection_scores = nullptr;
    const float *detection_classes = nullptr;

    // Map output layers by name
    for (auto const &layer : outputLayersInfo)
    {
        if (strcmp(layer.layerName, "num_detections") == 0)
        {
            num_detections = (const float *)layer.buffer;
        }
        else if (strcmp(layer.layerName, "detection_boxes") == 0)
        {
            detection_boxes = (const float *)layer.buffer;
        }
        else if (strcmp(layer.layerName, "detection_scores") == 0)
        {
            detection_scores = (const float *)layer.buffer;
        }
        else if (strcmp(layer.layerName, "detection_classes") == 0)
        {
            detection_classes = (const float *)layer.buffer;
        }
    }

    if (!num_detections || !detection_boxes || !detection_scores || !detection_classes)
    {
        std::cerr << "ERROR: Failed to find all required output layers." << std::endl;
        return false;
    }

    int num_dets = (int)num_detections[0]; // Assuming batch size 1 processed per call or first element is relevant if batched? 
    // Wait, NvDsInferParseCustomTriton receives one frame worth of data, but buffer points to the whole batch if batched?
    // Actually, `outputLayersInfo` usually contains pointers to the *current frame's* buffer offset by `NvDsInferContext`.
    // Let's assume standard behavior where `layer.buffer` points to the start of the tensor for THIS frame in the batch.
    
    // Safety check mostly for empty frames
    if (num_dets <= 0) return true;

    for (int i = 0; i < num_dets; ++i)
    {
        float score = detection_scores[i];
        if (score < detectionParams.perClassPreclusterThreshold[0]) // Global threshold or per class
            continue;
            
        // Get box coordinates
        // EfficientNMS usually outputs [x1, y1, x2, y2]
        float x1 = detection_boxes[i * 4 + 0];
        float y1 = detection_boxes[i * 4 + 1];
        float x2 = detection_boxes[i * 4 + 2];
        float y2 = detection_boxes[i * 4 + 3];
        
        // Clip to network dims
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min((float)networkInfo.width, x2);
        y2 = std::min((float)networkInfo.height, y2);

        if (x2 <= x1 || y2 <= y1) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = (unsigned int)detection_classes[i];
        obj.detectionConfidence = score;
        
        // IMPORTANT: Network uses input dims (640x640 usually). 
        // DeepStream expects coordinates relative to `networkInfo` resolution.
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;

        objectList.push_back(obj);
    }

    return true;
}

// Support strict interface check if needed
extern "C"
bool checkCustomParseFuncPrototype(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                   NvDsInferNetworkInfo const &networkInfo,
                                   NvDsInferParseDetectionParams const &detectionParams,
                                   std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return true; // Not strictly needed for runtime loading usually, but good practice
}
