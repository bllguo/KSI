import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader


def collate(block, word_to_ix):
    block_size = len(block)
    max_words = np.max([len(i[0]) for i in block])
    mat = np.zeros((block_size, max_words), dtype=int)
    for i in range(block_size):
        for j in range(max_words):
            try:
                if block[i][0][j] in word_to_ix:
                    mat[i,j] = word_to_ix[block[i][0][j]]
            except IndexError:
                pass
    mat = torch.from_numpy(mat)
    embeddings = torch.FloatTensor(np.array([x for _, x, _ in block]))
    labels = torch.FloatTensor(np.array([y for _, _, y in block]))
    return mat, embeddings, labels


def load_KSI_data(dir='data/original/',
                  batch_size=32,
                  train=True,
                  val=True,
                  test=True,
                  device='cpu'):
    training_data=np.load(f'{dir}training_data.npy', allow_pickle=True)
    test_data=np.load(f'{dir}test_data.npy', allow_pickle=True)
    val_data=np.load(f'{dir}val_data.npy', allow_pickle=True)
    word_to_ix=np.load(f'{dir}word_to_ix.npy', allow_pickle=True).item() # words (in notes) to index
    wikivec=np.load(f'{dir}newwikivec.npy', allow_pickle=True) # wiki article embeddings (# codes with wiki articles, vocab size)

    wikivec = torch.FloatTensor(wikivec).to(device)

    collate_fn = partial(collate, word_to_ix=word_to_ix)
    loaders = {}
    if train:
        loaders['train'] = DataLoader(training_data, collate_fn=collate_fn, batch_size=batch_size)
    if val:
        loaders['val'] = DataLoader(val_data, collate_fn=collate_fn, batch_size=batch_size)
    if test:
        loaders['test'] = DataLoader(test_data, collate_fn=collate_fn, batch_size=batch_size)

    return loaders, wikivec, word_to_ix


def train(model,
          dataloader,
          loss_function,
          wikivec=None,
          optimizer=None,
          profiler=None,
          scheduler=None,
          device='cpu',
          init_hidden=False):
    model.train()
    for data in dataloader:
        optimizer.zero_grad()
        note, embeddings, labels = data
        if init_hidden:
            model.hidden = model.init_hidden(len(note), device=device)
        note = note.to(device)
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        scores = model(note, embeddings, wikivec)
        loss = loss_function(scores, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if profiler:
            profiler.step()


def test_model(model,
               dataloader,
               wikivec=None,
               threshold=0.5,
               k=10,
               by_label=False,
               device='cpu',
               init_hidden=False):
    y = []
    yhat = []
    recall = []
    model.eval()
    for data in dataloader:
        note, embeddings, labels = data
        if init_hidden:
            model.hidden = model.init_hidden(len(note), device=device)
        note = note.to(device)
        embeddings = embeddings.to(device)
        out = model(note, embeddings, wikivec).cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        y.append(labels)
        yhat.append(out)

    y = np.concatenate(y)
    yhat = np.concatenate(yhat)
    preds = np.array(yhat > threshold, dtype=float)
    for i in range(yhat.shape[0]):
        n_labels = int(y[i, :].sum())
        topk = max(k, n_labels)
        ind_topk = np.argpartition(yhat[i, :], -topk)[-topk:]
        recall.append(y[i, ind_topk].sum() / n_labels if n_labels > 0 else np.nan)

    # compute macro AUC by label frequency group
    label_freq_aucs = None
    if by_label:
        loaded_bin_data = np.load('data/bin_data.npy', allow_pickle=True)
        bin_10 = loaded_bin_data[0]
        bin_50 = loaded_bin_data[1]
        bin_100 = loaded_bin_data[2]
        bin_500 = loaded_bin_data[3]
        bin_remaining = [4]
        label_freq_aucs = {}
        label_freq_aucs['1-10'] = roc_auc_score(y[:, bin_10], yhat[:, bin_10], average='macro')
        label_freq_aucs['11-50'] = roc_auc_score(y[:, bin_50], yhat[:, bin_50], average='macro')
        label_freq_aucs['51-100'] = roc_auc_score(y[:, bin_100], yhat[:, bin_100], average='macro')
        label_freq_aucs['101-500'] = roc_auc_score(y[:, bin_500], yhat[:, bin_500], average='macro')
        label_freq_aucs['>500'] = roc_auc_score(y[:, bin_remaining], yhat[:, bin_remaining], average='macro')

    # compute overall metrics
    mask = np.sum(y, axis=0) > 0 # mask out classes without both positive and negative examples
    recall = np.nanmean(recall)
    micro_f1 = f1_score(y[:, mask], preds[:, mask], average='micro')
    macro_f1 = f1_score(y[:, mask], preds[:, mask], average='macro')
    micro_auc = roc_auc_score(y[:, mask], yhat[:, mask], average='micro')
    macro_auc = roc_auc_score(y[:, mask], yhat[:, mask], average='macro')
    return recall, micro_f1, macro_f1, micro_auc, macro_auc, label_freq_aucs


def train_model(model,
                train_dataloader,
                val_dataloader,
                wikivec=None,
                optimizer=None,
                scheduler=None,
                n_epochs=10,
                profile=False,
                log_path='./log',
                device='cpu',
                init_hidden=False):
    loss_function = nn.BCELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    if profile:
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ], profile_memory=True,
               use_cuda=device != 'cpu',
               on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path)) as prof:
            for epoch in range(n_epochs):
                train(model,
                      train_dataloader,
                      loss_function,
                      wikivec=wikivec,
                      optimizer=optimizer,
                      profiler=prof,
                      scheduler=scheduler,
                      device=device,
                      init_hidden=init_hidden)
                t_recall_at_k, t_micro_f1, t_macro_f1, t_micro_auc, t_macro_auc, _ = test_model(model,
                                                                                                train_dataloader,
                                                                                                wikivec,
                                                                                                device=device,
                                                                                                init_hidden=init_hidden)
                v_recall_at_k, v_micro_f1, v_macro_f1, v_micro_auc, v_macro_auc, _ = test_model(model,
                                                                                                val_dataloader,
                                                                                                wikivec,
                                                                                                device=device,
                                                                                                init_hidden=init_hidden)
                print(f'Epoch: {epoch+1:03d}, Train Recall@10: {t_recall_at_k:.4f}, Val Recall@10: {v_recall_at_k:.4f}' +
                    f', Train Micro F1: {t_micro_f1:.4f}, Val Micro F1: {v_micro_f1:.4f}' +
                    f', Train Macro F1: {t_macro_f1:.4f}, Val Macro F1: {v_macro_f1:.4f}' +
                    f', Train Micro AUC: {t_micro_auc:.4f}, Val Micro AUC: {v_micro_auc:.4f}' +
                    f', Train Macro AUC: {t_macro_auc:.4f}, Val Macro AUC: {v_macro_auc:.4f}')
    else:
        prof = None
        for epoch in range(n_epochs):
            train(model,
                  train_dataloader,
                  loss_function,
                  wikivec=wikivec,
                  optimizer=optimizer,
                  profiler=prof,
                  scheduler=scheduler,
                  device=device,
                  init_hidden=init_hidden)
            t_recall_at_k, t_micro_f1, t_macro_f1, t_micro_auc, t_macro_auc, _ = test_model(model,
                                                                                            train_dataloader,
                                                                                            wikivec,
                                                                                            device=device,
                                                                                            init_hidden=init_hidden)
            v_recall_at_k, v_micro_f1, v_macro_f1, v_micro_auc, v_macro_auc, _ = test_model(model,
                                                                                            val_dataloader,
                                                                                            wikivec,
                                                                                            device=device,
                                                                                            init_hidden=init_hidden)
            print(f'Epoch: {epoch+1:03d}, Train Recall@10: {t_recall_at_k:.4f}, Val Recall@10: {v_recall_at_k:.4f}' +
                f', Train Micro F1: {t_micro_f1:.4f}, Val Micro F1: {v_micro_f1:.4f}' +
                f', Train Macro F1: {t_macro_f1:.4f}, Val Macro F1: {v_macro_f1:.4f}' +
                f', Train Micro AUC: {t_micro_auc:.4f}, Val Micro AUC: {v_micro_auc:.4f}' +
                f', Train Macro AUC: {t_macro_auc:.4f}, Val Macro AUC: {v_macro_auc:.4f}')
    return prof
