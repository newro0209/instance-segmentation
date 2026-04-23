import argparse
import yaml
import torch
from transformers import (
    TrainingArguments,
    set_seed
)
from instance_segmentation.models import load_segmentation_runtime
from instance_segmentation.utils.config_loader import resolve_config_path


def _get_config_section(config: dict[str, object], section_name: str) -> dict[str, object]:
    section = config.get(section_name)
    if isinstance(section, dict):
        return section
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    # Load configuration
    config_path = resolve_config_path(args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    training_config = _get_config_section(config, 'training')
    dataset_config = _get_config_section(config, 'dataset')

    set_seed(int(training_config.get('seed', 42)))

    # 1. Load Model & Processor
    runtime = load_segmentation_runtime(config)
    image_processor = runtime.image_processor
    model = runtime.model

    # 2. Load Dataset (Example)
    # dataset = load_dataset(config['dataset']['name'])

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(training_config.get('output_dir', './results')),
        num_train_epochs=int(training_config.get('num_epochs', 50)),
        per_device_train_batch_size=int(dataset_config.get('batch_size', 1)),
        learning_rate=float(training_config.get('learning_rate', 5e-5)),
        weight_decay=float(training_config.get('weight_decay', 0.0)),
        lr_scheduler_type=str(training_config.get('lr_scheduler_type', 'linear')),
        warmup_steps=int(training_config.get('warmup_steps', 0)),
        logging_steps=int(training_config.get('logging_steps', 10)),
        evaluation_strategy="steps",
        eval_steps=int(training_config.get('eval_steps', 500)),
        save_strategy="steps",
        save_steps=int(training_config.get('save_steps', 1000)),
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="tensorboard", # or "wandb"
    )

    # 4. Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset[config['dataset']['train_split']],
    #     eval_dataset=dataset[config['dataset']['test_split']],
    #     tokenizer=image_processor,
    # )

    # 5. Train
    print("Training setup complete. Ready to train.")
    # trainer.train()

if __name__ == "__main__":
    main()
