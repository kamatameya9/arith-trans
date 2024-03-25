import wandb
import argparse
# import itertools
import torch
import tqdm
import pickle
from collections import Counter
import torch.nn.functional as F

import dataset as my_datasets
from model import AdditionModel


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
        "--acc-next", type=float, default=0.95, help="Accuracy before next level"
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
    parser.add_argument(
        "--initial-number-length",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--preferred-dtype",
        type=str,
        default='int64',
        help="Use this dtype if possible (int64, object)"
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

    run = wandb.init(project="arith-trans", config=args)

    dataset = make_dataset(args, number_length=args.initial_number_length)

    model = AdditionModel(
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

    if args.compile:
        model = torch.compile(model)
    
    wandb.watch(model, log_freq=10)
    manual_training(model, dataset, args)


def make_dataset(args, number_length=1):
    kvargs = dict(
        preferred_dtype=args.preferred_dtype,
        base=args.base,
        number_length=number_length,
        pre_end_padding=args.cot_padding,
        flip=args.flip,
    )

    if args.op == "add":
        return my_datasets.BinaryOpDataset(
            func=(lambda a, b: a + b),
            sep="+",
            out_length=number_length + 1,
            **kvargs,
        )
    

def answer_mask(dataset, batch):
    """Creates a mask of everything after the END (or =) token, which separates the question
    from the answer."""
    mask = torch.cumsum(batch == dataset.end_token, dim=1) == 1
    mask &= batch != dataset.end_token
    return mask[:, 1:]


def training_step(model, batch):
    """Computes cross entropy loss between the model output and the ground truth, but only on
    the tokens after the END token, since the previous data is just random."""
    mask = answer_mask(model.ds, batch)
    truth = batch[:, 1:]
    out = model(batch)[:, :-1]
    return F.cross_entropy(out[mask], truth[mask])


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

    # We'd like to test that our validation method matches what you get with generate.
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


def manual_training(model, dataset, args):
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    batch_size = args.batch_size
    optimizer = model.configure_optimizers()

    # Standard PyTorch Training Loop
    # Note: Saving the train and val datasets for each number length. 
    # Keeping them the same accross epochs
    time_to_success = Counter()
    train_batches = args.train_batches
    val_batches = args.val_batches
    with torch.no_grad():
        np_data, train_carry = dataset.generate_batch(batch_size * train_batches)
        train_data = torch.tensor(np_data).to(device)
        np_data, val_carry = dataset.generate_batch(batch_size * val_batches)
        val_data = torch.tensor(np_data).to(device)
        with open(f'data/train_dataset_{dataset.number_length}.pickle', 'wb') as output:
            pickle.dump(train_data, output)
        with open(f'data/train_carry_{dataset.number_length}_mult.pickle', 'wb') as output:
            pickle.dump(train_carry, output)
        with open(f'data/val_dataset_{dataset.number_length}.pickle', 'wb') as output:
            pickle.dump(val_data, output)
        with open(f'data/val_carry_{dataset.number_length}_mult.pickle', 'wb') as output:
            pickle.dump(val_carry, output)

    for epoch in range(args.epochs):
        # train_batches = args.train_batches
        # with torch.no_grad():
        #     np_data = dataset.generate_batch(batch_size * train_batches)
        #     train_data = torch.tensor(np_data).to(device)

        # Training Loop
        losses = []
        model.train()
        for batch_idx in tqdm.tqdm(range(train_batches)):
            batch = train_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            optimizer.zero_grad()
            loss = training_step(model, batch)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        loss = torch.mean(torch.tensor(losses))

        # Validation Loop
        accs = []
        model.eval()
        with torch.no_grad():
            # val_batches = args.val_batches
            # np_data = dataset.generate_batch(batch_size * train_batches)
            # val_data = torch.tensor(np_data).to(device)

            for batch_idx in tqdm.tqdm(range(val_batches)):
                batch = val_data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                acc = validation_step(model, batch)
                accs.append(acc)
        
        # torch.onnx.export(model, batch, f"ckpts/len{dataset.number_length}_hsize64_flip_samedata_multcarry.onnx")
        # wandb.save(f"ckpts/len{dataset.number_length}_hsize64_flip_samedata_multcarry.onnx")
        acc = torch.mean(torch.tensor(accs))
        print(f"Validation acc: {acc:.5}")

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_acc": acc,
            }
        )

        
        
        # Print some examples. Try to always include an example where the model is wrong.
        # But if the model is nearly perfect, don't bother, since we might search forever.
        model.print_examples(3, must_include_a_wrong=acc < args.acc_next)
        
        # Saving the model
        torch.save(model.state_dict(), f"ckpts/len{dataset.number_length}_hsize64_flip_samedata_multcarry.ckpt")
        
        time_to_success[dataset.number_length] += 1

        print("Epochs per digit:", sorted(time_to_success.items()))
        if acc > args.acc_next:
            break
            print(f"Switching to number length {dataset.number_length+1}")
            dataset = make_dataset(args, number_length=dataset.number_length + 1)
            model.ds = dataset
            # Generating dataset for next number_length
            with torch.no_grad():
                np_data, train_carry = dataset.generate_batch(batch_size * train_batches)
                train_data = torch.tensor(np_data).to(device)
                np_data, val_carry = dataset.generate_batch(batch_size * val_batches)
                val_data = torch.tensor(np_data).to(device)
                with open(f'data/train_dataset_{dataset.number_length}.pickle', 'wb') as output:
                    pickle.dump(train_data, output)
                with open(f'data/val_dataset_{dataset.number_length}.pickle', 'wb') as output:
                    pickle.dump(val_data, output)
                with open(f'data/train_carry_{dataset.number_length}_mult.pickle', 'wb') as output:
                    pickle.dump(train_carry, output)
                with open(f'data/val_carry_{dataset.number_length}_mult.pickle', 'wb') as output:
                    pickle.dump(val_carry, output)

if __name__ == "__main__":
    main()
