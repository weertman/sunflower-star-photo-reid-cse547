"""
Example settings.txt file contents:

Pretrained: True
Model Version: densenet121
Dataset: Pycnopodia_helianthoides_full_event-unaware_training-split_2024-05-03-18-13
Embedding Size: 256
Number of Layers: 3
Batch Size: 8
Image Size: 256x256
Margin: 0.5
Learning Rate: 0.001
Number of Epochs: 15
Unfreeze Epoch: 15
Unfreeze All Epoch: 25
Number of Unfreeze Layers: 128
Optimizer: sgd
Momentum: 0.9
Scheduler Step Size: 1
Scheduler Gamma: 0.5
Number of Workers: 2
Prefetch Factor: 2

"""


# load settings
# put them in dataframe
# with loss
# and directory name
from glob import glob
import pandas as pd
import os
import os.path as op
import datetime
import matplotlib.pyplot as plt

date_txt_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
sweep_kind = 'full_event-unaware_training-split'
sweep_name = f"{sweep_kind}-{date_txt_now}"
sweep_dir = op.join('..', 'evaluations', 'learning', sweep_name)
if not op.exists(sweep_dir):
    print(f"Creating directory {sweep_dir}")
    os.makedirs(sweep_dir)
else:
    print(f"Directory {sweep_dir} already exists")

csv_path = op.join(sweep_dir, f"sweep_res.csv")

folders = glob((
    op.join("..", "models", "reid",
       f"*{sweep_kind}"
        "_2024-*-*-*-*__2024-*-*-*-*-*__*__densenet121__PT-True")))
print(f'Found {len(folders)} folders')

epochs = 150

dl = []
for j, folder_name in enumerate(folders):
    #print(folder_name)
    #print(f'\t{op.basename(folder_name).split("full_event")[0][:-1]}')
    this_row = {}
    try:
        with open(op.join(folder_name, "logs", "settings.txt"), 'r') as file:
            for line in file:
                key, value = map(str.strip, line.split(':', 1))
                this_row[key] = value
        ## check that Number of Epochs is epochs
        if int(this_row['Number of Epochs']) != epochs:
            #print(f"Skipping: Number of epochs is not {epochs}")
            continue


        for sett in ["train", "test"]:
            logs = pd.read_csv(op.join(
                folder_name, "logs", f"{sett}_learning_logs.csv"))

            epoch_loss = logs.groupby('epoch')['loss'].agg(['mean']).reset_index()
            ## calculate different between first and last epoch loss
            start_loss = epoch_loss[epoch_loss.epoch == 0]["mean"].to_numpy().flatten()[0]
            end_loss = epoch_loss[epoch_loss.epoch == epochs - 1]["mean"].to_numpy().flatten()[0]
            n = 0
            while end_loss is None:
                n += 1
                end_loss = epoch_loss[epoch_loss.epoch == epochs - 1 - n]["mean"].to_numpy().flatten()[0]
            diff = end_loss - start_loss

            for ii in range(epochs):
                try:
                    this_row[f"{sett}loss-{ii}"] = epoch_loss[
                        epoch_loss.epoch == ii]["mean"].to_numpy().flatten()[0]
                except IndexError:
                    this_row[f"{sett}loss-{ii}"] = None

            this_row[f"{sett}loss-diff"] = diff

    except IndexError as e:
        print(f"Error: {e} for {folder_name}")
        continue
    this_row["Dataset"] = op.basename(folder_name).split("full_event")[0][:-1]
    this_row["Model Folder Name"] = op.basename(folder_name)
    dl.append(this_row)

df = pd.DataFrame(dl)
df.to_csv(csv_path, index=False)

print(df.columns)

## sort df by test loss diff
df = df.sort_values(by=f"testloss-{epochs-1}", ascending=False)

margin_options = sorted(df["Margin"].unique())
batch_size_options = sorted(df["Batch Size"].unique())
optimizer_options = sorted(df["Optimizer"].unique())
embedding_dim_options = sorted(df["Embedding Size"].unique())
n_layers_embedding_options = sorted(df["Number of Layers"].unique())

def get_epoch_from_losscol(col):
    return int(col.split("-")[-1])

## get train loss columns
train_loss_columns = [col for col in df.columns if "trainloss" in col]
train_loss_columns = [col for col in train_loss_columns if "diff" not in col]
train_loss_columns = sorted(train_loss_columns, key=get_epoch_from_losscol)
print(f'Train loss columns: {train_loss_columns}')

## get test loss columns
test_loss_columns = [col for col in df.columns if "testloss" in col]
test_loss_columns = [col for col in test_loss_columns if "diff" not in col]
test_loss_columns = sorted(test_loss_columns, key=get_epoch_from_losscol)
print(f'Test loss columns: {test_loss_columns}')

fig, ax = plt.subplots(1, 1, figsize=(15,10))

path_fig = op.join(sweep_dir, "diagnostic_loss_plot.png")

train_color = "blue"
test_color = "orange"

xticks = list(range(epochs))
xlabs = [str(x+1) for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")

for ii, row in df.iterrows():
    train_loss = [row[col] for col in train_loss_columns]
    test_loss = [row[col] for col in test_loss_columns]

    ax.plot(train_loss, color=train_color, alpha=0.5)
    ax.plot(test_loss, color=test_color, alpha=0.5)

    optimizer = row["Optimizer"]
    margin = row["Margin"]
    batch_size = row["Batch Size"]
    dataset = row["Dataset"]
    num_layers = row["Number of Layers"]
    embedding_size = row["Embedding Size"]

    train_loss_diff = row["trainloss-diff"]
    test_loss_diff = row["testloss-diff"]

    text = f"{epochs} Opt: {optimizer}, Margin: {margin}, Batch: {batch_size}, Nlayers: {num_layers}, Emb. Size: {embedding_size}, Dataset: {dataset}"
    print(text)
    print(f'\tTrain loss diff: {train_loss_diff}, Test loss diff: {test_loss_diff}')
    print(f'\tTrain loss: {train_loss[-1]}, Test loss: {test_loss[-1]}')
    ax.text(epochs-1, train_loss[-1], text, color=train_color, alpha=0.5)
    ax.text(epochs-1, test_loss[-1], text, color=test_color, alpha=0.5)

## turn off top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim([0, epochs+10])

fig.tight_layout()

plt.savefig(path_fig, dpi=300, bbox_inches='tight')

plt.show()
plt.close()

