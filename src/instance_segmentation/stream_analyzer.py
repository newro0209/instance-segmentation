import argparse
import yaml
import torch
import cv2
import numpy as np
from transformers import AutoModelForInstanceSegmentation, AutoImageProcessor
from PIL import Image

def list_cameras(max_tested=5):
    """사용 가능한 카메라 인덱스 목록을 출력합니다."""
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml (selects the model)")
    parser.add_argument("--camera_id", type=int, default=0, help="ID of the camera to use")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    # 1. Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config['model']['name']
    
    print(f"[*] Loading model: {model_name} on {device}...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForInstanceSegmentation.from_pretrained(model_name).to(device)
    model.eval()

    # 2. Setup Camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera with ID {args.camera_id}")
        cameras = list_cameras()
        print(f"[*] Available camera IDs: {cameras}")
        return

    print(f"[*] Camera {args.camera_id} started. Press 'q' to quit.")

    # 3. Visualization loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert OpenCV BGR to RGB for Model
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Prepare Inputs
        inputs = image_processor(images=pil_image, return_tensors="pt").to(device)

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-processing
        target_sizes = [(frame.shape[0], frame.shape[1])]
        results = image_processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]

        # Draw results on frame
        overlay = frame.copy()
        
        for score, label, mask in zip(results["scores"], results["labels"], results["masks"]):
            if score > args.threshold:
                mask = mask.cpu().numpy().astype(np.uint8) * 255
                color = np.random.randint(0, 255, (3,)).tolist()
                
                # Draw Mask overlay
                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask > 0] = color
                cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0, overlay)
                
                # Draw Label and Score
                label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
                # Find mask contour to place text
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    cv2.putText(overlay, label_text, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show frame
        cv2.imshow(f"Instance Segmentation - {config['model']['type']}", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
