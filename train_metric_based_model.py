from torch.utils import data
import torch
import torch.nn
from logging import getLogger
from util import logger as L
import sys
from time import time
from loss import attn_loss
import argparse

from sas import handling_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", "-tr")
    parser.add_argument("--dev_dir", "-dv")
    parser.add_argument("--test_dir", "-ts")
    parser.add_argument("--OUT_DIR", "-o")
    parser.add_argument("--BERT_BASE_DIR")
    parser.add_argument("--print-attention", "-pa", dest="print_attention", default=False,
                        action='store_true', help="")
    parser.add_argument("--model-path", "-mp", dest="model_path",
                        default=None, help="")
    parser.add_argument("--print-hidden_states", "-ph",
                        dest="print_hidden_states", default=False,
                        action='store_true', help="")

    parser.add_argument("-satt", dest="satt", default=False,
                        action='store_true', help="")

    parser.add_argument("--BERT", "-BERT", dest="BERT", default=False,
                        action='store_true', help="")

    parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>',
                        default=300, help="RNN dimension.")
    parser.add_argument("-emb_dim", "--emb_dim", dest="emb_dim", type=int, metavar='<int>',
                        default=100, help="RNN dimension.")
    parser.add_argument("-v_size", "--vocab_size", dest="vocab_size", type=int, metavar='<int>',
                        default=4000, help="satt loss ratio")
    parser.add_argument("-l", "--lamda", dest="lamda", type=float, metavar='<float>',
                        default=1.0, help="satt loss ratio")
    parser.add_argument("--optimizer","-opt", dest="opt", default='rmsprop',choices=['rmsprop','sgd','adagrad','adadelta','adam'], help="")



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
    parser.add_argument("--update-embedding", "-ue", dest="update_embed", default=False,
                        action='store_true', help="")

    parser.add_argument("--dummy", "-dmy", dest="dmy", default=False,
                        action='store_true', help="")

    parser.add_argument("--print-ids", dest="pids", default=False,
                        action='store_true', help="")

    parser.add_argument("--regression", "--reg", dest="reg", default=False,
                        action='store_true', help="")


    parser.add_argument("--MoT", "-mot", dest="mot", default=False,
                        action='store_true', help="")
    parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                        help="The path to the word embeddings file (Word2Vec format)")

    parser.add_argument("--improved", "-imp", dest="imp", default=False,
                        action='store_true', help="")
    parser.add_argument("--TrustScore-loss", "-tsl", dest="tsl", default=False,
                        action='store_true', help="")
    parser.add_argument("--ranked-triplet-loss", "-rtl", dest="rtl", default=False,
                        action='store_true', help="")
    parser.add_argument("--ranked-triplet-loss-normalized", "-rtln", dest="rtln", default=False,
                        action='store_true', help="")
    parser.add_argument("--qudratic-ranked-triplet-loss-normalized", "-qrtln", dest="qrtln", default=False,
                        action='store_true', help="")
    parser.add_argument("--dual-ranked-triplet-loss-normalized", "-drln", dest="drln", default=False,
                        action='store_true', help="")
    parser.add_argument("-m", "--margin", dest="margin", type=float, metavar='<float>',
                        default=0.2, help="satt loss ratio")

    args = parser.parse_args()

    return args


def main():
    # from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    from transformers import BertConfig
    import getinfo

    args = parse_args()
    info = getinfo.TrainInfo(args)

    L.set_logger(out_dir=info.out_dir, debug=args.debug)
    logger = getLogger(__name__)
    logger.info(f"{sys.argv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from model import BertForPartScoringWithLSTM, BiRnnModel

    import part_scoring_info as psi

    prompt = psi.prompt_check(info.train_dir)[0]
    logger.info(f"Prompt name :{prompt}")
    if args.BERT:
        from sas.input_data import input_data_ps_BERT as input_data
    else:
        from sas.input_data import input_data_ps as input_data

    (train_x, train_y, train_mask, train_attention), (dev_x, dev_y,
                                                      dev_mask), (test_x, test_y, test_mask) = input_data(info)


    info.store_part_scoring_info()
    info.device = device
    info.train_size = train_x.shape[0]
    logger.debug(f"factor name :{info.factors}")
    logger.debug(f"ps_labels:{info.ps_labels}")

    if args.BERT:
        if args.char:
            config = BertConfig.from_pretrained("cl-tohoku/bert-base-japanese-char-whole-word-masking")
        else:
            config = BertConfig.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
        model = BertForPartScoringWithLSTM(config, info=info)
        model.freeze_bert_pram()
    else:
        model = BiRnnModel(info)

    if args.model_path != None:
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Load pretrained model from {args.model_path}")
    # model.unfreeze_bert_layer_param(-1)

    model.to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion_satt = torch.nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    from sas.optimizer import get_optimizer
    optimizer = get_optimizer(info.opt,model)

    from metric_learning.triplet_loss import OnlineTripletLoss, ImprovedOnlineTripletLoss, TrustScoreOnlineTripletLoss,RankedTripletLoss,NomalizedRankedTripletLoss,NormalizedDualOnlineTripletLoss
    if args.imp:
        logger.info(f"Loss: Improved Triplet Loss")
        triplet_loss = ImprovedOnlineTripletLoss(0.2, 0.08)
    elif args.tsl:
        logger.info(f"Loss: TrustScore Triplet Loss")
        triplet_loss = TrustScoreOnlineTripletLoss(0.8)
    elif args.rtln:
        logger.info(f"Loss: Normalized Ranked Triplet Loss")
        triplet_loss = NomalizedRankedTripletLoss(args.margin,p=1)
    elif args.qrtln:
        logger.info(f"Loss: Normalized Quadratic Ranked Triplet Loss")
        triplet_loss = NomalizedRankedTripletLoss(args.margin,p=2)
    elif args.drln:
        logger.info(f"Loss: Normalized Dual Ranked Triplet Loss")
        triplet_loss = NormalizedDualOnlineTripletLoss(args.margin,p=1)
    else:
        logger.info(f"Loss: Triplet Loss")
        triplet_loss = OnlineTripletLoss(args.margin)

    logger.info(f"Optimizer: {optimizer}")

    from Dataset import part_scoring_set
    train_shape = train_x.shape

    dataset = part_scoring_set(train_x, train_y, train_mask, train_attention)

    dataloader = data.DataLoader(
        dataset, **{'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0})

    eval_dataloader = data.DataLoader(
        dataset, **{'batch_size': 400, 'shuffle': False, 'num_workers': 0})

    best = {"Hol": -1.0}
    best_mses = {"Hol": (1.0,1.0)}

    test_best = -1
    total_train_time = 0

    logger_train = getLogger('Training')
    logger_eval = getLogger('Evaluation')
    import evaluator_metric

    logger.debug(f"train shape:{train_x.shape}")
    logger.debug(f"dev shape: {dev_x.shape}")
    logger.debug(f"test shape:{test_x.shape}")

    eval = evaluator_metric.Evaluator(
        info, dev_x.to(device), test_x.to(device), dev_y, test_y, dev_mask.to(device), test_mask.to(device))

    for ii in range(args.epochs):
        b_loss = list()
        t0 = time()
        model.train()

        train_hidden_states = [torch.zeros(
            train_shape[0], info.rnn_dim) for k in info.ps_labels]
        train_golds = [torch.zeros(
            train_shape[0]) for k in info.ps_labels]

        index = 0
        for num, (batch_x, batch_y, batch_m, batch_a) in enumerate(dataloader):
            mini_batch_size = batch_y[0].shape[0]
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)

            outputs = model(batch_x,
                            token_type_ids=None, attention_mask=batch_m)

            output = outputs[0]
            attn_output = outputs[1]
            hidden_states = outputs[2][0]

            assert len(output) == len(info.ps_labels), logger.error(
                f"output len:{len(output)} != ps_labels len: {len(info.ps_labels)}")


            loss = 0
            loss_satt = 0
            t_loss = 0
            triplet_num = 0

            for i, label in enumerate(info.ps_labels):

                batch_y_ps = batch_y[i +1].type(torch.LongTensor).abs().to(device)
                train_hidden_states[i][index:index + batch_x.shape[0], :] = hidden_states[i]
                train_golds[i][index:index + batch_x.shape[0]] = batch_y[i + 1]

               # tripletが作れルト気時
                if len(torch.unique(batch_y_ps)) != 1:
                    triplet_l = triplet_loss(hidden_states[i], batch_y_ps,label[1])
                    if triplet_l[1] != 0:
                        t_loss += triplet_l[0]
                        triplet_num += triplet_l[1]


                if i < len(info.main_factors) and info.satt:
                    batch_a_ps = batch_a[i].to(device)
                    p_len = attn_output[i].shape[1]

                    loss_satt += attn_loss(attn_output[i],
                                          batch_a_ps[:, :p_len])

            loss += t_loss

            if info.satt:
                loss += loss_satt * args.lamda
            b_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += batch_x.shape[0]


        tr_time = time() - t0
        total_train_time += tr_time

        logger.info(
            f"Epoch:{ii} Train time :{tr_time:.2f} (s) Number of total triplet: {triplet_num}")
        logger.info(
            f"Scoring loss: {t_loss.item()/len(dataloader):.3f}")
        logger.info(
            f"loss: {sum(b_loss)/len(b_loss) / len(info.ps_labels):.3f}")
        model.eval()

        with torch.no_grad():
            (dev_qwks, test_qwks, dev_mses, test_mses), (dev_attentions, dev_hidden_states), (
                test_attentions, test_hidden_states), d_pred, t_pred, dev_target, test_target, (d_dist, t_dist), (d_ev, t_ev) = eval.evaluate(info, model, eval_dataloader)

            # logger_eval.info(f"[Dev]: {dev_qwk:.3f} [Test]: {test_qwk:.3f}")
            if best["Hol"] < dev_qwks["Hol"]:
                best = dev_qwks
                best_mses = dev_mses
                test_best = test_qwks
                test_best_mses = test_mses
                b_epoch = ii
                best_test_attentions = test_attentions
                best_test_hidden_states = test_hidden_states
                best_dev_attentions = dev_attentions
                best_dev_hidden_states = dev_hidden_states
                best_dev_pred = d_pred
                best_test_pred = t_pred
                best_d_dist = d_dist
                best_t_dist = t_dist
                best_d_ev = d_ev
                best_t_ev = t_ev
                best_model_weight = model.state_dict()
                handling_data.save_hidden_states_metric(
                    info, dev_hidden_states[0], d_pred, dev_target, d_dist, best_d_ev, prefix="dev")
                handling_data.save_hidden_states_metric(
                    info, test_hidden_states[0], t_pred, test_target, t_dist, best_t_ev, prefix="test")
                if not args.not_save_weights:
                    torch.save({
                        'epoch': ii,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, info.out_dir+"_best_checkpoint.pt")

        # print(pred_eval)

    logger.info(f"Best epoch {b_epoch}")
    eval.print_info(best, best_mses, type="Dev", log=logger)
    eval.print_info(test_best, test_best_mses, type="Test", log=logger)

    # logger.info(
    #    f"BEST -> [Dev]: {best:.3f} [Test]: {test_best:.3f} ({b_epoch} epoch)")

    handling_data.save_result_part_scoring_metric(
        info, best_test_pred, test_mask.sum(1),data_path=info.test_dir, probs=True, attentions=best_test_attentions, hidden_states=best_test_hidden_states, distance=best_t_dist, evidence=best_t_ev, prefix="test")
    handling_data.save_result_part_scoring_metric(
        info, best_dev_pred, dev_mask.sum(1), data_path = info.dev_dir,probs=True, attentions=best_dev_attentions, hidden_states=best_dev_hidden_states, distance=best_d_dist, evidence=best_d_ev, prefix="dev")
    model.load_state_dict(torch.load(info.out_dir +"_best_checkpoint.pt")['model_state_dict'])
    model.eval()
    handling_data.save_train_result_part_scoring_cls(info, model, device, eval_dataloader)


    import cloudpickle
    with open(info.out_dir + '_train_info.pickle', mode='wb') as f:
        cloudpickle.dump(info, f)


if __name__ == '__main__':
    main()
