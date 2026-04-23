import argparse
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoModelForInstanceSegmentation, AutoImageProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_name = args.checkpoint if args.checkpoint else config['model']['name']
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    
    # 1. Load Processor and Model
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForInstanceSegmentation.from_pretrained(model_name).to(device)

    # 2. Prepare Image
    image = Image.open(args.image).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # 3. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Post-processing (모델 타입에 따라 처리가 다를 수 있음)
    # Hugging Face AutoModel은 공통된 인터페이스를 제공하지만, 
    # 모델별 특화된 후처리는 image_processor.post_process_instance_segmentation을 사용 권장
    target_sizes = [image.size[::-1]]
    results = image_processor.post_process_instance_segmentation(outputs, target_sizes=target_sizes)[0]

    # 5. Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for score, label, mask in zip(results["scores"], results["labels"], results["masks"]):
        if score > args.threshold:
            mask = mask.cpu().numpy()
            # 간단한 마스크 시각화 로직
            plt.imshow(mask, alpha=0.4)
            print(f"Detected {model.config.id2label[label.item()]} with score {score:.3f}")

    plt.axis("off")
    plt.title(f"Model: {config['model']['type']}")
    plt.show()

if __name__ == "__main__":
    main()
