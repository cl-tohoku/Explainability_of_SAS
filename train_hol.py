import argparse
from torch.utils import data
import torch
import torch.nn
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from logging import setLoggerClass, getLogger
from util import logger as L
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", "-tr")
    parser.add_argument("--dev_dir", "-dv")
    parser.add_argument("--test_dir", "-ts")

    parser.add_argument("--OUT_DIR", "-o")
    parser.add_argument("--BERT_BASE_DIR")
    parser.add_argument("--print-attention", "-pa", dest="print_attention", default=False,
                        action='store_true', help="")
    parser.add_argument("--print-hidden_states", "-ph",
                        dest="print_hidden_states", default=False,
                        action='store_true', help="")

    parser.add_argument("-satt", dest="satt", default=False,
                        action='store_true', help="")

    parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>',
                        default=300, help="RNN dimension.")

    parser.add_argument("--epochs", dest="epochs", type=int,
                        metavar='<int>', default=50, help="Number of epochs (default=50)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int,
                        metavar='<int>', default=32, help="Batch size (default=32)")
    parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5,
                        help="T/he dropout probability. To disable, give a negative number (default=0.5)")
    parser.add_argument("--bidirectional", "-bi", dest="bidirectional", default=False,
                        action='store_true', help="")

    parser.add_argument("--print-debug-info", "-debug", dest="debug", default=False,
                        action='store_true', help="")
    parser.add_argument("--char-base", "-char", dest="char", default=False,
                        action='store_true', help="")

    parser.add_argument("--not-save-weights", "-nsw", dest="not_save_weights", default=False,
                        action='store_true', help="")

    parser.add_argument("--dummy", dest="dmy", default=False,
                        action='store_true', help="")

    parser.add_argument("--regression", "--reg", dest="reg", default=False,
                        action='store_true', help="")

    parser.add_argument("--MoT", "-mot", dest="mot", default=False,
                        action='store_true', help="")

    parser.add_argument("--print-ids", dest="pids", default=False,
                        action='store_true', help="")
    args = parser.parse_args()

    return args


def main():
    import getinfo
    import part_scoring_info as psi

    args = parse_args()
    info = getinfo.TrainInfo(args)

    L.set_logger(out_dir=info.out_dir, debug=args.debug)
    logger = getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info.store_device(device)
    from model import BertForScoring, BertForScoringWithLSTM

    prompt = psi.prompt_check(info.train_dir)[0]
    ranges = psi.get_ps_ranges()
    info.high = ranges["all"][prompt][1]
    info.low = ranges["all"][prompt][0]

    info.labels = info.high + 1



    logger.info(f"Load pretrained bert model from {info.BERT_BASE_DIR}")
    model = BertForScoringWithLSTM.from_pretrained(
        info.BERT_BASE_DIR, num_labels=info.labels, output_hidden_states=True, t_info=info)

    model.freeze_bert_pram()
    # model.unfreeze_bert_layer_param(-2)
    model.to(device)

    from sas.input_data import input_data
    import sas.input_data as in_d

    (train_x, train_y, train_mask), (dev_x, dev_y,
                                     dev_mask), (test_x, test_y, test_mask) = input_data(info)

    if args.reg:
        train_y = in_d.convert_to_model_friendlly_score(info, train_y)

    # info.store_part_scoring_info()

    logger.debug(train_x.shape)
    logger.debug(dev_x.shape)
    logger.debug(test_x.shape)

    if args.reg:
        criterion = torch.nn.MSELoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), 2e-6)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=0.001, alpha=0.9, eps=1e-6)

    logger.info(f"Optimizer: {optimizer}")
    from Dataset import hol_scoring_set
    dataset = hol_scoring_set(train_x, train_y, train_mask)

    dataloader = data.DataLoader(
        dataset, **{'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0})

    eval_dataloader = data.DataLoader(
        dataset, **{'batch_size': 400, 'shuffle': False, 'num_workers': 0})

    best = 0
    test_best = 0
    total_train_time = 0

    logger_train = getLogger('Train')
    logger_eval = getLogger('Evaluation')

    for i in range(args.epochs):
        b_loss = list()
        t0 = time()

        for batch_x, batch_y, batch_m in dataloader:
            model.train()

            batch_x = batch_x.to(device)

            if info.reg:
                batch_y = batch_y.to(device)
            else:
                batch_y = batch_y.type(torch.LongTensor).to(device)

            batch_m = batch_m.to(device)

            logits, _, _ = model(batch_x,
                                 token_type_ids=None, attention_mask=batch_m)

            # logger.debug(torch.round(logits * info.high))
            # logger.debug(torch.round(batch_y * info.high))

            loss = criterion(logits, batch_y)
            b_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tr_time = time() - t0
        total_train_time += tr_time

        logger_train.info(f"Epoch:{i} Train time :{tr_time:.2f}(s)")

        logger_train.info(f"loss: {sum(b_loss)/len(b_loss):.3f}")
        model.eval()

        with torch.no_grad():
            t_logits, test_attentions, test_hidden_states = model(test_x.to(device), labels=test_y.to(
                device), token_type_ids=None, attention_mask=test_mask.to(device))
            if args.reg:
                t_pred = t_logits.detach().cpu() * info.high
                test_qwk = qwk(torch.round(t_pred), test_y.cpu())
                t_probs = None

            else:
                t_pred = t_logits.softmax(1).argmax(1).detach().cpu()
                t_probs = t_logits.softmax(1).detach().cpu().numpy()

                test_qwk = qwk(t_pred, test_y.cpu())

            d_logits, dev_attentions, dev_hidden_states = model(dev_x.to(device), labels=dev_y.to(device),
                                                                token_type_ids=None, attention_mask=dev_mask.to(device))

            if args.reg:
                d_pred = d_logits.squeeze().detach().cpu() * info.high
                dev_qwk = qwk(torch.round(d_pred.clone()), dev_y.cpu())
                d_probs = None

            else:
                d_pred = d_logits.softmax(1).argmax(1).detach().cpu()
                d_probs = d_logits.softmax(1).detach().cpu().numpy()
                dev_qwk = qwk(d_pred, dev_y.cpu())

            for k, (batch_x, batch_y, batch_m) in enumerate(eval_dataloader):
                if k == 0:
                    train_logits, train_attention, train_hidden_states = model(batch_x.to(device), labels=batch_y.to(
                        device), token_type_ids=None, attention_mask=batch_m.to(device))

                    if info.reg:
                        train_pred = train_logits.squeeze().detach().cpu() * info.high
                    else:
                        train_pred = train_logits.softmax(
                            1).argmax(1).detach().cpu()
                    train_hidden_states = train_hidden_states[0]
                    continue

                train_logits_, train_attention_, train_hidden_states_ = model(batch_x.to(device), labels=batch_y.to(
                    device), token_type_ids=None, attention_mask=batch_m.to(device))

                if info.reg:
                    train_pred_ = (train_logits_.squeeze() *
                                   info.high).detach().cpu()
                else:
                    train_pred_ = train_logits_.softmax(
                        1).argmax(1).detach().cpu()
                train_pred = torch.cat((train_pred, train_pred_))
                train_hidden_states = torch.cat(
                    (train_hidden_states, train_hidden_states_[0]))

            if best < dev_qwk:
                best = dev_qwk
                test_best = test_qwk
                b_epoch = i
                best_dev_probs = d_probs
                best_test_attentions = test_attentions
                best_test_hidden_states = test_hidden_states
                best_train_attentions = train_attention
                best_train_hidden_states = train_hidden_states
                best_dev_attentions = dev_attentions
                best_dev_hidden_states = dev_hidden_states
                best_test_probs = t_probs
                best_dev_pred = d_pred
                best_test_pred = t_pred
                best_train_pred = train_pred
                if not args.not_save_weights:
                    torch.save(model.state_dict(), info.out_dir +
                               "_best_model_weights.pt")

        logger_eval.info(f"[Dev]: {dev_qwk:.3f} [Test]: {test_qwk:.3f}")

        # print(pred_eval)
    logger.info(
        f"BEST -> [Dev]: {best:.3f} [Test]: {test_best:.3f}({b_epoch} epoch)")

    import sas.handling_data as handling_data

    if args.reg:
        handling_data.save_result_cls(
            info, best_dev_pred, probs=best_dev_probs, attentions=None, hidden_states=best_dev_hidden_states, prefix="dev", out_probs=False)
        handling_data.save_result_cls(
            info, best_test_pred, probs=best_test_probs, attentions=None, hidden_states=best_test_hidden_states, out_probs=False)
        handling_data.save_result_cls(
            info, best_train_pred, probs=None, attentions=None, hidden_states=(best_train_hidden_states, None), prefix="train")
    else:
        handling_data.save_result_cls(
            info, best_dev_pred, probs=best_dev_probs, attentions=None, hidden_states=best_dev_hidden_states, prefix="dev", out_probs=True)
        handling_data.save_result_cls(
            info, best_test_pred, probs=best_test_probs, attentions=None, hidden_states=best_test_hidden_states, out_probs=True)
        handling_data.save_result_cls(
            info, best_train_pred, probs=None, attentions=None, hidden_states=(best_train_hidden_states, None), prefix="train")


if __name__ == '__main__':
    main()
