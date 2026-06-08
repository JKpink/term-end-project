"""Run experiments: train or evaluate collaborative agents."""
import os
import sys
import argparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
sys.path.insert(0, SRC_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", nargs="?", default="train",
                        choices=["train", "eval", "all", "demo"])
    parser.add_argument("--model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", default="./outputs/collab")
    parser.add_argument("--lora_path", default="./outputs/collab")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    if args.action in ("train", "all"):
        from train import train_collaborative
        train_collaborative(
            model_name=args.model_name,
            output_dir=args.output_dir,
            dataset_size=320,
            num_epochs=3,
            lora_r=8,
        )

    if args.action in ("eval", "all"):
        from evaluate import run_evaluation
        run_evaluation(
            model_name=args.model_name,
            lora_path=args.lora_path,
            num_samples=args.num_samples,
        )

    if args.action == "demo":
        from app.gradio_app import create_demo
        demo = create_demo(args.lora_path)
        demo.launch(server_name="0.0.0.0")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
