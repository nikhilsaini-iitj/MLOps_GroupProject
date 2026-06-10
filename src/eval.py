"""Evaluation, metrics, and W&B artifact logging.

Group Project — MLOps End-to-End Pipeline
"""
import json
import wandb
from sklearn.metrics import classification_report

from utils import compute_metrics


def evaluate_model(trainer, test_dataset, id2label, output_path="eval_report.json"):
    """Run evaluation, log metrics to W&B, save classification report as artifact."""
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # Log final metrics explicitly
    wandb.log({
        "final/loss": eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1": eval_results["eval_f1"],
    })

    # Save full classification report
    preds = trainer.predict(test_dataset).predictions.argmax(-1)
    labels = [item["labels"].item() for item in test_dataset]

    report = classification_report(
        labels,
        preds,
        target_names=list(id2label.values()),
        output_dict=True,
    )

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Upload to W&B as a versioned artifact
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(output_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    return eval_results


if __name__ == "__main__":
    pass
