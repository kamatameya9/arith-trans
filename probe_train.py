import os
import time
import argparse
# import itertools
import torch
import tqdm
import pickle
# from collections import Counter
import torch.nn.functional as F
from torch.utils.data import Dataset
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import numpy as np

import dataset_v2 as my_datasets
from model import AdditionModel, AdditionModelforProbing
from probe_model import BatteryProbeClassification, BatteryProbeClassificationTwoLayer
from probe_trainer import Trainer, TrainerConfig


def main():
    # Needed to enable tensor cores
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of examples to generate and train on",
    )
    parser.add_argument("--train-batches", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam LR")
    parser.add_argument(
        "--acc-next", type=float, default=0.9, help="Accuracy before next level"
    )
    # 0.05:  [(1, 1), (2, 2), (3, 8), (4, 11), (5, 25), (6, 76), (7, 206+)]
    # 0.04:  [(1, 1), (2, 2), (3, 8), (4, 7),  (5, 17), (6, 44), (7, 142), (8, 80+)]
    # 0.03:  [(1, 1), (2, 2), (3, 9), (4, 7),  (5, 11), (6, 32), (7, 121), (8, 110+)]
    # 0.02:  [(1, 1), (2, 2), (3, 8), (4, 8),  (5, 18), (6, 29), (7, 105), (8, 118+)]
    # 0.015: [(1, 1), (2, 2), (3, 8), (4, 16), (5, 25), (6, 70), (7, 131), (8, 107)] 
    # 0.01:  [(1, 1), (2, 2), (3, 8), (4, 12), (5, 14), (6, 44), (7, 151), (8, 341), (9, 151)]
    #        [(1, 1), (2, 2), (3, 7), (4, 15), (5, 107), (6, 247)]
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="The hidden size for the neural network",
    )
    parser.add_argument(
        "--ffw-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="The number of layers for the neural network",
    )
    parser.add_argument("--batch-size", type=int, default=2**10, help="Batch size")
    parser.add_argument(
        "--kind",
        required=True,
        type=str,
        help="The type of neural network to use (lstm, transformer, hybrid)",
    )
    parser.add_argument(
        "--op",
        type=str,
        default="add",
        help="Operation to learn (add, mult)",
    )
    parser.add_argument(
        "--cot-padding",
        type=int,
        default=0,
        help="Chain of thought padding",
    )
    parser.add_argument(
        "--base",
        type=int,
        default=10,
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--flip", action="store_true", help="Flip order of numbers")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="The number of heads/rank in transformer/mlp",
    )
    args = parser.parse_args()

    dataset = make_dataset(args, number_length=3)

    model = AdditionModelforProbing(
        ds=dataset,
        kind=args.kind,
        hidden_size=args.hidden_size,
        ffw_size=2 * args.hidden_size if args.ffw_size is None else args.ffw_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        dropout=args.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} parameters")
    load_res = model.load_state_dict(torch.load(f"./ckpts/len3_hsize64_flip_samedata_multcarry.ckpt"))
    if args.compile:
        model = torch.compile(model)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    model.train()
    with torch.no_grad():
        with open(f'data/train_dataset_{dataset.number_length}.pickle', 'rb') as data:
            train_data = pickle.load(data)
        with open(f'data/train_carry_{dataset.number_length}_mult.pickle', 'rb') as data:
            train_carry = pickle.load(data)

    act_container = []
    property_container = []
    for batch_idx in tqdm.tqdm(range(len(train_data))):
        batch = train_data[batch_idx:batch_idx+1]
        # print(batch)
        ind = (batch[0]==dataset.end_token).nonzero()[0][0]
        # print(ind)
        end = (batch[0]==dataset.eos_token).nonzero()[0][0]
        print(end)
        print(int(ind))
        act = model(batch)[0, ...].detach().cpu()[ind:end]#[:dataset.number_length+1]
        print(len(act), "act")
        act_container.extend(act)
        properties = train_carry[batch_idx][:len(act)]
        # print(len(properties))
        property_container.extend(properties)

    age_container = []
    for batch_idx in tqdm.tqdm(range(len(train_data))):
        batch = train_data[batch_idx:batch_idx+1]
        ind = (batch[0]==dataset.end_token).nonzero()[0][0]
        end = (batch[0]==dataset.eos_token).nonzero()[0][0]
        act = model(batch)[0, ...].detach().cpu()[ind:end][:4]
        act_container.extend(act)
        properties = train_carry[batch_idx][:len(act)]
        property_container.extend(properties)

    # for x, y in tqdm(loader, total=len(loader)):
    #     tbf = [train_dataset.itos[_] for _ in x.tolist()[0]]
    #     valid_until = tbf.index(-100) if -100 in tbf else 999
    #     a = OthelloBoardState()
    #     ages = a.get_gt(tbf[:valid_until], "get_age")  # [block_size, ]
    #     age_container.extend(ages)

    with torch.no_grad():
        with open(f'data/acts_{dataset.number_length}.pickle', 'wb') as output:
            pickle.dump(act_container, output)
        with open(f'data/properties_{dataset.number_length}.pickle', 'wb') as output:
            pickle.dump(property_container, output)
        # with open(f'data/acts_{dataset.number_length}.pickle', 'rb') as data:
        #     act_container = pickle.load(data)
        # with open(f'data/properties_{dataset.number_length}.pickle', 'rb') as data:
        #     property_container = pickle.load(data)

    probe = BatteryProbeClassification(device, probe_class=2, num_task=dataset.number_length+1, input_dim=64)
    # probe = BatteryProbeClassificationTwoLayer(device, probe_class=2, num_task=dataset.number_length, mid_dim=16)

    class ProbingDataset(Dataset):
        def __init__(self, act, y):
            assert len(act) == len(y)
            print(f"{len(act)} pairs loaded...")
            self.act = act
            self.y = y
            print(np.sum(np.array(y)==0), np.sum(np.array(y)==1))
            
        def __len__(self, ):
            return len(self.y)
        def __getitem__(self, idx):
            return self.act[idx], torch.tensor(self.y[idx]).to(torch.long)
        
    probing_dataset = ProbingDataset(act_container, property_container)    
    # print(probing_dataset[0][0].shape, probing_dataset[0][1].shape, probing_dataset[0][2].shape)
    # print(probing_dataset[0])
    train_size = int(0.8 * len(probing_dataset))
    test_size = len(probing_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(probing_dataset, [train_size, test_size])
    # sampler = None
    # train_loader = DataLoader(train_dataset, shuffle=False, sampler=sampler, pin_memory=True, batch_size=128, num_workers=1)
    # test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=128, num_workers=1)

    max_epochs = 16
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=1024, learning_rate=1e-3,
        betas=(.9, .999), 
        lr_decay=True, warmup_tokens=len(train_dataset)*5, 
        final_tokens=len(train_dataset)*max_epochs,
        num_workers=4, weight_decay=0., 
        ckpt_path=os.path.join("./ckpts/", f"{dataset.number_length}_1l",
        num_task=dataset.number_length+1)
    )
    trainer = Trainer(probe, train_dataset, test_dataset, tconf)
    trainer.train(prt=True)
    trainer.save_traces()
    trainer.save_checkpoint()


def make_dataset(args, number_length=1):
    kvargs = dict(
        base=args.base,
        number_length=number_length,
        pre_end_padding=args.cot_padding,
        flip=args.flip,
    )
    if args.op == "addmod":
        return my_datasets.AddModDataset(**kvargs)
    elif args.op == "divmod":
        return my_datasets.DivModDataset(**kvargs)
    elif args.op == "add":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a + b),
            sep="+",
            out_length=number_length + 1,
            **kvargs,
        )
    elif args.op == "mult":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a * b),
            sep="*",
            out_length=2 * number_length,
            **kvargs,
        )
    elif args.op == "div":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a // b),
            sep="//",
            min_b=1,
            out_length=number_length,
            **kvargs,
        )
    elif args.op == "mod":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a % b),
            sep="%",
            min_b=1,
            out_length=number_length,
            **kvargs,
        )
    elif args.op == "sqmod":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a**2 % b),
            sep="^2 %",
            min_b=1,
            out_length=2 * number_length,
            **kvargs,
        )
    elif args.op == "factor":
        return my_datasets.FactorDataset(**kvargs)


def answer_mask(dataset, batch):
    """Creates a mask of everything after the END (or =) token, which separates the question
    from the answer."""
    mask = torch.cumsum(batch == dataset.end_token, dim=1) == 1
    mask &= batch != dataset.end_token
    return mask[:, 1:]


# def training_step(model, batch):
#     """Computes cross entropy loss between the model output and the ground truth, but only on
#     the tokens after the END token, since the previous data is just random."""
#     mask = answer_mask(model.ds, batch)
#     truth = batch[:, 1:]
#     out = model(batch)[:, :-1]
#     return F.cross_entropy(out[mask], truth[mask])


def validation_step(model, batch):
    """Computes the accuracy on the model, if we assume greedy decoding is used.
    We only consider a question corectly solved if every single token is correctly predicted,
    including the padding."""
    mask = answer_mask(model.ds, batch)
    truth = batch[:, 1:]
    out = model(batch)[:, :-1]
    preds = torch.argmax(out, dim=2)

    # print(f'{truth[0]=}')
    # print(f'{preds[0]=}')

    # We'd to test that our validation method matches what you get with generate.
    # Unfortunately the LSTMs give slightly different results when passing a batch,
    # vs when passing one element at a time, which breaks the direct correspondance.
    for i in range(0):
        n = batch[i].tolist().index(model.ds.end_token) + 1
        true = batch[i, n:]
        pred0 = preds[i, n - 1 :]
        pred1 = model.generate(batch[i][:n])
        if torch.all((preds * mask)[i] == (truth * mask)[i]):
            assert torch.all(pred0 == true)
            # If we are getting the answer right, they should be the same.
            assert torch.all(pred0 == pred1)
        else:
            # If we are getting the answer wrong, they should both be wrong.
            assert not torch.all(pred0 == true)
            assert not torch.all(pred1 == true)

    return torch.all(preds * mask == truth * mask, dim=1).float().mean()


if __name__ == "__main__":
    main()
