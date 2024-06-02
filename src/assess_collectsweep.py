import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    path_sweeps = os.path.join('..', 'evaluations', '*', 'sweep_res.csv')
    sweeps = glob.glob(path_sweeps)
    ## get last created
    sweep = max(sweeps, key=os.path.getctime)
    print(sweep)

    plot_dir = os.path.join(os.path.dirname(sweep), 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    results_dir = os.path.join(os.path.dirname(sweep), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df = pd.read_csv(sweep)
    df = df[df['Optimizer'] != 'adam']

    print(df.head())
    print(df.columns)

    hyper_params = ['Optimizer',
                    'Margin',
                    'Batch Size',
                    'Number of Layers',
                    'Embedding Size']

    datasets = sorted(df['Dataset'].unique())

    train_loss_columns = [col for col in df.columns if "trainloss" in col]
    train_loss_columns = [col for col in train_loss_columns if "diff" not in col]

    test_loss_columns = [col for col in df.columns if "testloss" in col]
    test_loss_columns = [col for col in test_loss_columns if "diff" not in col]

    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        tmp = df[df['Dataset'] == dataset]

        ## for each hyper parameter category find its unique values
        hyper_dict = {}
        for hyper_param in hyper_params:
            print(f"{hyper_param} options: {sorted(tmp[hyper_param].unique())}")
            hyper_dict[hyper_param] = sorted(tmp[hyper_param].unique())

        ## for each hyper parameter combination calculate average train and test loss
        ## store into a dictionary

        fig, ax = plt.subplots(1, 1, figsize=(8,6))

        runs_dict = {}
        for optimizer in hyper_dict['Optimizer']:
            for margin in hyper_dict['Margin']:
                for batch_size in hyper_dict['Batch Size']:
                    for n_layers in hyper_dict['Number of Layers']:
                        for embedding_size in hyper_dict['Embedding Size']:

                            tmp2 = tmp[(tmp['Optimizer'] == optimizer) &
                                       (tmp['Margin'] == margin) &
                                       (tmp['Batch Size'] == batch_size) &
                                       (tmp['Number of Layers'] == n_layers) &
                                       (tmp['Embedding Size'] == embedding_size)]

                            if tmp2.shape[0] > 0:
                                ## average train and test loss over all runs
                                train_loss = tmp2[train_loss_columns].mean(axis=0)
                                test_loss = tmp2[test_loss_columns].mean(axis=0)
                                ## average train and test loss difference over all runs
                                train_loss_diff = tmp2['trainloss-diff'].mean()
                                test_loss_diff = tmp2['testloss-diff'].mean()

                                xvalues = np.arange(0, len(train_loss))

                                text = f"opt:{optimizer}, margin:{margin}, batch:{batch_size}, layers:{n_layers}, emb_dim:{embedding_size}"
                                ax.plot(xvalues, train_loss, linewidth=1, alpha=0.5, color='blue')
                                ax.plot(xvalues, test_loss, linewidth=1, alpha=0.5, color='orange')
                                ax.text(xvalues[-1]+1, test_loss[-1], text, color='orange', alpha=0.5)

                                runs_dict[(optimizer, margin, batch_size, n_layers, embedding_size)] = {
                                    'train_loss': train_loss,
                                    'test_loss': test_loss,
                                    'train_loss_diff': train_loss_diff,
                                    'test_loss_diff': test_loss_diff
                                }

        ax.set_title('Dataset: ' + dataset, loc='left')

        ax.set_xlim(0, len(xvalues)+100)
        ax.set_ylim(0, 0.5)

        ## turn off top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        path_fig = os.path.join(plot_dir, f"{dataset}_diagnostic_loss_plot.png")
        plt.savefig(path_fig, dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

        ## for each MARGIN setting find the best 3 and worst 1 performing runs
        ## based on the test loss
        ## store into a text file
        path_results = os.path.join(results_dir, f"{dataset}_results.txt")
        with open(path_results, 'w') as f:
            for margin in hyper_dict['Margin']:
                print(f"\n\nMargin: {margin}")
                f.write(f"\n\nMargin: {margin}\n")
                tmp3 = tmp[tmp['Margin'] == margin]

                ## sort by test loss
                tmp3 = tmp3.sort_values(by=test_loss_columns[-1])

                ## get best 3 and worst 1
                best_3 = tmp3.head(3)
                worst_1 = tmp3.tail(1)

                f.write(f"Best 3:\n")
                for ii, row in best_3.iterrows():
                    optimizer = row['Optimizer']
                    batch_size = row['Batch Size']
                    n_layers = row['Number of Layers']
                    embedding_size = row['Embedding Size']
                    train_loss_diff = row['trainloss-diff']
                    test_loss_diff = row['testloss-diff']
                    train_loss = row[train_loss_columns].values
                    test_loss = row[test_loss_columns].values

                    text = f"opt:{optimizer}, batch:{batch_size}, layers:{n_layers}, emb_dim:{embedding_size}"
                    print(text)
                    f.write(text + '\n')
                    print(f"\tTrain loss diff: {train_loss_diff}, Test loss diff: {test_loss_diff}")
                    f.write(f"\tTrain loss diff: {train_loss_diff}, Test loss diff: {test_loss_diff}\n")
                    print(f"\tTrain loss: {train_loss[-1]}, Test loss: {test_loss[-1]}")
                    f.write(f"\tTrain loss: {train_loss[-1]}, Test loss: {test_loss[-1]}\n")

                f.write(f"Worst 1:\n")
                for ii, row in worst_1.iterrows():
                    optimizer = row['Optimizer']
                    batch_size = row['Batch Size']
                    n_layers = row['Number of Layers']
                    embedding_size = row['Embedding Size']
                    train_loss_diff = row['trainloss-diff']
                    test_loss_diff = row['testloss-diff']
                    train_loss = row[train_loss_columns].values
                    test_loss = row[test_loss_columns].values

                    text = f"opt:{optimizer}, batch:{batch_size}, layers:{n_layers}, emb_dim:{embedding_size}"
                    print(text)
                    f.write(text + '\n')
                    print(f"\tTrain loss diff: {train_loss_diff}, Test loss diff: {test_loss_diff}")
                    f.write(f"\tTrain loss diff: {train_loss_diff}, Test loss diff: {test_loss_diff}\n")
                    print(f"\tTrain loss: {train_loss[-1]}, Test loss: {test_loss[-1]}")
                    f.write(f"\tTrain loss: {train_loss[-1]}, Test loss: {test_loss[-1]}\n")
