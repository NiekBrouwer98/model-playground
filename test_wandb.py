import wandb
import random

wandb_key = input('Enter your wandb key: ')

wandb.login(key=wandb_key, force=True)

wandb.init( project="my-awesome-project",
           config={"learning_rate": 0.02,
                   "architecture": "CNN",
                   "dataset": "CIFAR-100",
                   "epochs": 10})

epochs = 10
offset = random.random() / 5

for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()