import argparse
import warnings

import os
os.environ['TL_BACKEND'] = 'torch'

import numpy as np
import torch
# import torch.cuda as cuda
# import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
# from torch import optim
# from torch_geometric.nn import DeepGraphInfomax
import tensorlayerx as tlx
import paddle
import torch
import tensorflow as tf

# from GraphPAR.src.models.model import GNN, Classifier  # 确保GraphPAR是你的顶级目录
# from GraphPAR.src.utils import load_data, setup_seed, calculate_parameter_count, fair_metric, get_path_to
from GammaGL.gammagl.models.gnn import GNN, Adapter, Classifier
from GammaGL.gammagl.utils.process import load_data, setup_seed, calculate_parameter_count, fair_metric, get_path_to
from GammaGL.gammagl.models.deepgraphinfomax import DeepGraphInfomax


warnings.filterwarnings('ignore')


class WithLoss(tlx.nn.Module):
    def __init__(self, net, loss_fn):
        super(WithLoss, self).__init__()
        self._net = net
        self._loss_fn = loss_fn

    def forward(self, data, label):
        pos_z, neg_z, summary = self._net(data.x, data.edge_index)
        loss = self._loss_fn(pos_z, neg_z, summary)
        return loss

# 随机打乱节点特征x的顺序，但保持边的连接关系(edge_index)不变，用于对比学习中的负样本生成
def corruption(x, edge_index):
    indices = np.random.permutation(x.shape[0])
    return x[indices], edge_index


def train(params, data, pgm_file):
    PGM = GNN(data.input_dim, params.hidden_dim, params.num_layer, params.activation)
    # PGM = PGM.to(params.device)
    # print(PGM.trainable_weights)
    model = DeepGraphInfomax(hidden_channels=params.hidden_dim, encoder=PGM,
                             summary=lambda h, *args, **kwargs: tlx.sigmoid(tlx.reduce_mean(h, axis=0)), corruption=corruption)
    # if tlx.BACKEND == 'torch':
    #     optimizer = torch.optim.Adam(model.parameters(), lr=params.pre_lr, weight_decay=params.weight_decay)
    # else:
    optimizer = tlx.optimizers.Adam(lr=params.pre_lr, weight_decay=params.weight_decay)
    calculate_parameter_count(model)
    model = model.to(params.device)
    best_loss = 10000
    net_with_loss = WithLoss(model, loss_fn=model.loss)
    net_with_train = tlx.model.TrainOneStep(net_with_loss, optimizer, model.trainable_weights)
    for epoch in range(1, params.pre_epochs + 1):
        # model.train()
        model.set_train()
        if tlx.BACKEND == 'tensorflow':
            with tf.GradientTape() as tape:
                pos_z, neg_z, summary = model(data.x, data.edge_index)
                loss = model.loss(pos_z, neg_z, summary)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        elif tlx.BACKEND == 'torch':
            # # # for param in model.trainable_weights:
            # # #     if param.grad is not None:
            # # #         param.grad.zero_()
            # # optimizer.zero_grad()
            # # pos_z, neg_z, summary = model(data.x, data.edge_index)
            # # loss = model.loss(pos_z, neg_z, summary)
            # # loss.backward()
            # # optimizer.step()
            # pos_z, neg_z, summary = model(data.x, data.edge_index)
            # loss = model.loss(pos_z, neg_z, summary)
            # grads = optimizer.gradient(loss, model.trainable_weights)
            # # loss.backward()
            # # grads = [param.grad for param in model.trainable_weights]
            # optimizer.apply_gradients(zip(grads, model.trainable_weights))
            loss = net_with_train(data, None)
        
        elif tlx.BACKEND == 'paddle':
            pos_z, neg_z, summary = model(data.x, data.edge_index)
            loss = model.loss(pos_z, neg_z, summary)
            # grads = paddle.grad(loss, model.trainable_weights)
            grads = optimizer.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # print(PGM.trainable_weights)
        if loss < best_loss:
            model.eval()
            # tlx.files.save_weights_to_hdf5(PGM.trainable_weights, pgm_file)

            # weights = [(name, param) for name, param in PGM.state_dict().items()]
            # tlx.files.save_weights_to_hdf5(weights, pgm_file)

            # net.save_weights(args.best_model_path+net.name+".npz", format='npz_dict')
            PGM.save_weights(pgm_file, format='npz_dict')
            best_loss = loss
        if epoch % 100 == 0:
            print(f'Epoch {epoch}/{params.pre_epochs}, Loss: {loss:.4f}')

    

def loop(classifier, optimizer, embeddings, labels, sens, mode="train"):
    logits = classifier(embeddings)
    loss = tlx.losses.sigmoid_cross_entropy(logits, tlx.cast(tlx.expand_dims(labels, axis=1), dtype=tlx.float32))
    if mode == "train":
        if tlx.BACKEND == 'tensorflow':
            import tensorflow as tf
            with tf.GradientTape() as tape:
                loss = tlx.losses.sigmoid_cross_entropy(logits, tlx.cast(tlx.expand_dims(labels, axis=1), dtype=tlx.float32))
            grads = tape.gradient(loss, classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads, classifier.trainable_weights))
        elif tlx.BACKEND == 'torch':
            # for param in classifier.trainable_weights:
            #     if param.grad is not None:
            #         param.grad.zero_()
            # loss.backward()
            # grads = [param.grad for param in classifier.trainable_weights]
            # optimizer.apply_gradients(zip(grads, classifier.trainable_weights))
            grads = optimizer.gradient(loss, classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads, classifier.trainable_weights))
        elif tlx.BACKEND == 'paddle':
            import paddle
            grads = paddle.grad(loss, classifier.trainable_weights)
            optimizer.apply_gradients(zip(grads, classifier.trainable_weights))

    # predictions = (logits > 0).astype(labels.dtype)
    # acc = accuracy_score(y_true=labels.numpy(), y_pred=predictions.numpy())
    # f1 = f1_score(y_true=labels.numpy(), y_pred=predictions.numpy())
    # parity, equality = fair_metric(predictions.numpy(), labels.numpy(), sens.numpy())
    predictions = tlx.cast(logits > 0, dtype=labels.dtype)
    acc = accuracy_score(y_true=tlx.convert_to_numpy(labels), y_pred=tlx.convert_to_numpy(predictions))
    f1 = f1_score(y_true=tlx.convert_to_numpy(labels), y_pred=tlx.convert_to_numpy(predictions))
    parity, equality = fair_metric(tlx.convert_to_numpy(predictions), tlx.convert_to_numpy(labels), tlx.convert_to_numpy(sens))
    return acc * 100, f1 * 100, equality * 100, parity * 100, loss


def test_pgm(data, params, pgm_file):
    PGM = GNN(data.input_dim, params.hidden_dim, params.num_layer, params.activation).to(params.device)
    # tlx.files.load_hdf5_to_weights(pgm_file, PGM)
    PGM.load_weights(pgm_file, format='npz_dict')
    # PGM.eval()
    PGM.set_eval()
    pgm_embeddings = PGM(data.x, data.edge_index).detach()
    idx_train, idx_val, idx_test = data.idx_train_list, data.idx_valid_list, data.idx_test_list
    sens_train, sens_val, sens_test = data.sens[idx_train], data.sens[idx_val], data.sens[idx_test]
    train_embeddings, valid_embeddings, test_embeddings = pgm_embeddings[idx_train], pgm_embeddings[idx_val], \
        pgm_embeddings[idx_test]
    classifier = Classifier(input_dim=params.hidden_dim, num_classes=1).to(params.device)
    optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=params.weight_decay)
    best_perf = -1
    test_acc, test_f1, test_equality, test_parity = -1, -1, -1, -1
    for epoch in range(1000):
        # classifier.train()
        classifier.set_train()
        _ = loop(classifier, optimizer, train_embeddings, data.y[idx_train], sens_train)
        # classifier.eval()
        classifier.set_eval()
        valid_acc, valid_f1, valid_equality, valid_parity, valid_loss = loop(classifier, optimizer, valid_embeddings,
                                                                             data.y[idx_val], sens_val, mode="eval")
        valid_perf = (valid_acc + valid_f1)
        if valid_perf > best_perf and epoch > 20:
            best_perf = valid_perf
            test_acc, test_f1, test_equality, test_parity, _ = loop(classifier, optimizer, test_embeddings,
                                                                    data.y[idx_test], sens_test, mode="eval")
            print(
                f"Epoch:{epoch}, Acc:{valid_acc:.4f}, F1: {valid_f1:.4f}, DP: {valid_parity:.4f}, EO: {valid_equality:.4f}")
    print(f"Test Acc:{test_acc:.4f}, F1: {test_f1:.4f}, DP: {test_parity:.4f}, EO: {test_equality:.4f}")
    return test_acc, test_f1, test_equality, test_parity


def get_params():
    # device = tlx.BACKEND
    # if device == 'torch':
    #     import torch
    #     current_device = 0 if torch.cuda.is_available() else 'cpu'
    # elif device == 'tensorflow':
    #     import tensorflow as tf
    #     current_device = 0 if tf.config.list_physical_devices('GPU') else 'cpu'
    # elif device == 'paddle':
    #     import paddle
    #     current_device = 0 if paddle.get_device()=='gpu' else 'cpu'
    # else:
    #     current_device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--pre_train', type=str, default='infomax')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='pokec_z', choices=['income', 'credit', 'pokec_z', 'pokec_n'])
    parser.add_argument('--hidden_dim', type=int, default=24)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--pre_epochs', type=int, default=2000)
    parser.add_argument('--pre_lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--activation', type=str, default='leakyrelu', choices=['relu', 'leakyrelu', 'prelu'])
    return parser.parse_args()

def main():
    # params = get_params()
    # if params.device >= 0:
    #     tlx.set_device("GPU",params.device)
    # else:
    #     tlx.set_device("CPU")
    params = get_params()
    if params.device >= 0: 
        if tlx.BACKEND == "torch" and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            device = "CPU"
        elif tlx.BACKEND == "paddle" and not paddle.get_device()=='gpu':
            print("Paddle GPU is not available, switching to CPU.")
            device = "CPU"
        elif tlx.BACKEND == "tensorflow" and not tf.config.list_physical_devices('GPU'):
            print("TensorFlow GPU is not available, switching to CPU.")
            device = "CPU"
        else:
            device = 'GPU'
    else: 
        device = "CPU"
    tlx.set_device(device)
    # print(torch.cuda.is_available()) 
    # print(f"Running on {device}")
    # print(tlx.get_device())
    acc_array, f1_array, equality_array, parity_array = np.array([]), np.array([]), np.array([]), np.array([])
    seeds = [11, 13, 15, 17, 19]
    for seed in seeds:
        print("=" * 25, f"seed={seed}", "=" * 25)
        params.seed = seed
        setup_seed(params.seed)
        # tlx.set_seed(args.seed)
        print("Arguments: %s " % ",".join([("%s=%s" % (k, v)) for k, v in params.__dict__.items()]))
        file_name = (f"{params.dataset}_{params.seed}_{params.activation}_hidden-dim({params.hidden_dim})_"
                     f"num-layer({params.num_layer})_epochs({params.pre_epochs})_lr({params.pre_lr})_weight_decay({params.weight_decay})")
        model_path = get_path_to('saved_models')
        pgm_file = f'{model_path}/pretrain/infomax/{file_name}_weights.npz'
        print("Save the PGM to:", pgm_file)
        data = load_data(params)
        # print(data)
        train(params, data, pgm_file)
        acc, f1, equality, parity = test_pgm(data, params, pgm_file)
        acc_array = np.append(acc_array, acc)
        f1_array = np.append(f1_array, f1)
        equality_array = np.append(equality_array, equality)
        parity_array = np.append(parity_array, parity)
    print(f"Acc(↑):{acc_array.mean():.2f}±{acc_array.std():.2f}, F1(↑):{f1_array.mean():.2f}±{f1_array.std():.2f}, "
          f"DP(↓):{parity_array.mean():.2f}±{parity_array.std():.2f}, EO(↓):{equality_array.mean():.2f}±{equality_array.std():.2f}")


if __name__ == '__main__':
    main()
