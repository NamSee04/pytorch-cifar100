import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from tqdm import tqdm
import editdistance

from zc3.preprocess import test_loader, LABEL2CHAR, CHAR2LABEL
from models import c3_squeezenet
from zc3.ctc_decoder import ctc_decode

config = {
    'reload_checkpoint': 'checkpoints/c3/squeezenet.pt',
    'decode_method': 'greedy',
    'beam_size': 10,
}

torch.backends.cudnn.enabled = False


def evaluate(model, dataloader, criterion, max_iter=None, decode_method='greedy', beam_size=10):
    
    model.eval()
    tot_count = 0
    char_count = 0
    tot_loss = 0
    leven_dis = 0

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

            images, targets, target_lengths = [d.to(device) for d in data]

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()

            tot_loss += loss.item()
            tot_count += batch_size
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                for i in range(len(real)):
                    real[i] = chr(ord('a') + real[i] - 1)
                real = ''.join(real)


                for i in range(len(pred)):
                    pred[i] = chr(ord('a') + pred[i] - 1)
                pred = ''.join(pred)

                if real == 'zc':
                    char_count += 1
                    if pred != real:
                        leven_dis += 1
                else:
                    leven_dis += editdistance.eval(real, pred)
                    char_count += target_length

                target_length_counter += target_length
            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': 1 - leven_dis / char_count,
    }
    return evaluation


def main():
    reload_checkpoint = config['reload_checkpoint']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = c3_squeezenet()
    model.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    model.to(device)

    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    evaluation = evaluate(model, test_loader, criterion,
                          decode_method=config['decode_method'],
                          beam_size=config['beam_size'])
    print('test_evaluation: loss={loss}, acc={acc}'.format(**evaluation))


if __name__ == '__main__':
    main()