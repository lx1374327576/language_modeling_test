import torch


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, train_batch in enumerate(iterator):
        text, target = train_batch.text, train_batch.target
        predict, _ = model(text)
        train_loss = criterion(predict.view(-1, predict.shape[-1]), target.view(-1))

        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += train_loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, eval_batch in enumerate(iterator):
            text, target = eval_batch.text, eval_batch.target
            predict, _ = model(text)
            eval_loss = criterion(predict.view(-1, predict.shape[-1]), target.view(-1))

            epoch_loss += eval_loss.item()

    return epoch_loss / len(iterator)
