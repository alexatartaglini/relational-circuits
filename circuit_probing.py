import os
import shutil
import uuid
import copy

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

import utils
import evals


class TemperatureCallback:
    # A simple callback that updates the probes temperature parameter,
    # which transforms a soft mask into a hard mask
    def __init__(self, total_epochs, final_temp):
        self.temp_increase = final_temp ** (1.0 / total_epochs)

    def update(self, model):
        temp = model.temperature
        model.temperature = temp * self.temp_increase


def eval_probe(config, probe, dataloader):
    # Implements a simple probe evaluation loop
    probe.train(False)
    average_eval_loss = []
    for batch in dataloader:
        batch = {k: v.to(config["device"]) for k, v in batch.items()}
        out = probe(**batch)
        average_eval_loss.append(out.loss.detach().item())
    probe.train(True)
    return np.sum(average_eval_loss) / len(average_eval_loss)


def train_probe(config, probe, trainloader):
    # Implements a simple training loop that optimizes binary masks over networks
    temp_callback = TemperatureCallback(config["num_epochs"], config["max_temp"])
    optimizer = AdamW(probe.parameters(), lr=config["lr"])
    num_training_steps = len(trainloader) * config["num_epochs"]
    progress_bar = tqdm(range(num_training_steps))

    probe.train()

    for name, param in probe.named_parameters():
        if hasattr(param, "requires_grad") and param.requires_grad == True:
            print(name)
             
    for epoch in range(config["num_epochs"]):
        progress_bar.set_description(f"Training epoch {epoch}")
        average_train_loss = []
        for batch in trainloader:
            batch = {k: v.to(config["device"]) for k, v in batch.items()}
            out = probe(**batch)
            loss = out.loss
            loss.backward()
            average_train_loss.append(loss.detach().item())
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        train_loss = np.sum(average_train_loss) / len(average_train_loss)
        l0_statistics = probe.wrapped_model.wrapped_model.compute_l0_statistics()

        progress_bar.set_postfix(
            {
                "Train Loss": round(train_loss, 4),
                "L0 Norm": l0_statistics["total_l0"].detach().item(),
                "L0 Max": l0_statistics["max_l0"],
            }
        )

        temp_callback.update(probe.wrapped_model.wrapped_model)

    return probe


def main():
    config = utils.get_config()
    df = pd.DataFrame()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config["device"] = device
    shutil.rmtree(config["results_dir"], ignore_errors=True)
    os.makedirs(config["results_dir"])

    if config["save_models"]:
        shutil.rmtree(config["model_dir"], ignore_errors=True)
        os.makedirs(os.path.join(config["model_dir"]), exist_ok=True)

    # Iterate through all training hyperparameters
    for lr in config["lr_list"]:
        for batch_size in config["batch_size_list"]:
            for target_layer in config["target_layer_list"]:
                for operation in config["operation_list"]:
                    for mask_init in config["mask_init_list"]:
                        for seed in config["seed_list"]:
                            # Create a new model_id
                            model_id = str(uuid.uuid4())

                            config["lr"] = lr
                            config["batch_size"] = batch_size
                            config["target_layer"] = target_layer
                            config["operation"] = operation
                            config["mask_init_value"] = mask_init
                            config["seed"] = seed

                            # Set seed
                            torch.manual_seed(seed)
                            np.random.seed(seed)
                            print(config)
                            model, transform = utils.get_model(config)
                            trainloader, testloader = utils.create_datasets(config, transform)

                            probe = utils.create_circuit_probe(config, model)
                            probe.to(device)

                            # Get full model KNN results
                            probe.wrapped_model.wrapped_model.use_masks(False)
                            pre_knn_results = evals.knn_evaluation(
                                config, probe, trainloader, testloader, per_class=1
                            )
                            probe.wrapped_model.wrapped_model.use_masks(True)

                            if config["num_epochs"] != 0:
                                probe = train_probe(config, probe, trainloader)
                            final_train_loss = eval_probe(config, probe, trainloader)
                            final_eval_loss = eval_probe(config, probe, testloader)

                            # Run KNN subnetwork evaluation
                            knn_results = evals.knn_evaluation(
                                config, probe, trainloader, testloader, per_class=1
                            )

                            l0_statistics = (
                                probe.wrapped_model.wrapped_model.compute_l0_statistics()
                            )

                            output_dict = {
                                "model_id": [model_id],
                                "Variable": [config["variable"]],
                                "batch_size": [config["batch_size"]],
                                "lr": [config["lr"]],
                                "seed": [seed],
                                "target_layer": [config["target_layer"]],
                                "operation": [config["operation"]],
                                "mask_init": [mask_init],
                                "model dir": [config["model_dir"]],
                                "train loss": [final_train_loss],
                                "test loss": [final_eval_loss],
                                "knn dev acc": [knn_results["dev_acc"]],
                                "dev majority acc": [knn_results["dev_majority"]],
                                "knn test acc": [knn_results["test_acc"]],
                                "test majority acc": [knn_results["test_majority"]],
                                "full model knn dev acc": [pre_knn_results["dev_acc"]],
                                "full model knn test acc": [
                                    pre_knn_results["test_acc"]
                                ],
                                "L0 Norm": [l0_statistics["total_l0"].cpu().item()],
                                "L0 Max": [l0_statistics["max_l0"]],
                            }

                            if config["sd_eval"] == True:
                                # Run on various data splits
                                splits = [{1: "same"}, {0: "different"}, {0: "different-color"}, {0: "different-shape"}, {0: "different-texture"}]
                                for split in splits:
                                    # Run Same-Different evaluation to see the effect of ablating subnetworks
                                    # Extract underlying ViT Model
                                    split_name = list(split.values())[0]
                                    probe.train(False)
                                    model = probe.wrapped_model.wrapped_model
                                    sd_loader = utils.get_sd_data(config, split, transform)

                                    # Ablate subnetworks and run SD-eval
                                    model.set_ablate_mode("zero_ablate")
                                    sd_results = evals.same_diff_eval(config, model, sd_loader)

                                    output_dict[f"{split_name} vanilla acc"] = [
                                        sd_results["vanilla_acc"]
                                    ]
                                    output_dict[f"{split_name} ablated acc"] = [
                                        sd_results["ablated_acc"]
                                    ]

                                    # Ablate random subnetworks and rerun evaluation
                                    if (
                                        config["num_epochs"] != 0
                                        and config["num_random_ablations"] != 0
                                    ):
                                        # Can configure N random samples/reruns
                                        random_ablated_accs = []
                                        for _ in range(config["num_random_ablations"]):
                                            model.set_ablate_mode(
                                                "complement_sampled",
                                                force_resample=True,
                                            )

                                            try:
                                                # Try to run complement_sampled ablation
                                                # If discovered mask doesn't allow for this,
                                                # consider it a failure and return -1
                                                # Complement sampled ablations are samples
                                                # from the complement of the discovered mask
                                                random_ablate_sd_results = evals.same_diff_eval(
                                                    config, model, sd_loader
                                                )
                                            except:
                                                random_ablate_sd_results = {
                                                    "ablated_acc": [-1]
                                                }

                                            random_ablated_accs.append(
                                                random_ablate_sd_results["ablated_acc"]
                                            )

                                        output_dict[f"{split_name} random ablate acc mean"] = [
                                            np.mean(random_ablated_accs)
                                        ]
                                        output_dict[f"{split_name} random ablate acc std"] = [
                                            np.std(random_ablated_accs)
                                        ]

                            model.set_ablate_mode("none")

                            df = pd.concat(
                                [df, pd.DataFrame.from_dict(output_dict)],
                                ignore_index=True,
                            )

                            print("Saving csv")
                            # Will overwrite this file after every evaluation
                            df.to_csv(
                                os.path.join(config["results_dir"], "results.csv")
                            )

                            if config["save_models"]:
                                torch.save(
                                    probe.state_dict(),
                                    os.path.join(config["model_dir"], model_id + ".pt"),
                                )


if __name__ == "__main__":
    main()
