from torch.utils import data
import torch
import torch.nn
import torch.nn.functional as F
from logging import getLogger
from util import logger as L
import sys
from time import time
from loss import attn_loss, modified_attn_loss
import argparse
from sas import handling_data, util
import pandas as pd
import numpy as np
import plotly.express as px
from util.save_norm import Norm
from util.save_token_embedding import TokenDict
from collections import defaultdict
import cloudpickle


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
    parser.add_argument("--optimizer", "-opt", dest="opt", default='rmsprop',
                        choices=['rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam'], help="")

    parser.add_argument("--neputune", "-npt", dest="npt", type=str,
                        help="run with neptune.ai. ex. -npt user_name/project_name experiment_name", nargs=2)

    parser.add_argument("--epochs", dest="epochs", type=int,
                        metavar='<int>', default=50, help="Number of epochs (default=50)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int,
                        metavar='<int>', default=32, help="Batch size (default=32)")
    # parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.0,
    #                     help="T/he dropout probability. To disable, give a negative number (default=0.5)")
    parser.add_argument("--dropout_prob", dest="dropout_prob", type=float, metavar='<float>', default=0,
                        help="T/he dropout probability. To disable, give a negative number (default=0.5)")
    parser.add_argument("--rnn_dropout", type=float, default=0,
                        help="T/he dropout probability. To disable, give a negative number (default=0.5)")

    parser.add_argument("--bidirectional", "-bi", dest="bidirectional", default=False,
                        action='store_true', help="")
    parser.add_argument("--lstm_num", '-ln', type=int,
                        default=1, help="number of lstm layer")
    parser.add_argument("-gru", action="store_true", default=False,
                        help='use gru instead of lstm')
    parser.add_argument("--bilstm_process", '-bp', type=str, choices=[
                        "sum", "mean", "concat"], default="sum", help="way to process bilstm layer vec")
    parser.add_argument("--print-debug-info", "-debug", dest="debug", default=False,
                        action='store_true', help="")
    parser.add_argument("--char-base", "-char", dest="char", default=False,
                        action='store_true', help="")
    parser.add_argument("--affin", "-affin", dest="affin", default=False,
                        action='store_true', help="use linear layer")
    parser.add_argument("--not-save-weights", "-nsw", dest="not_save_weights", default=False,
                        action='store_true', help="")
    # parser.add_argument("--update-embedding", "-ue", dest="update_embed", default=False,
    #                     action='store_true', help="")

    parser.add_argument("--dummy", "-dmy", dest="dmy", default=False,
                        action='store_true', help="")

    parser.add_argument("--item", "-item", dest="item")

    parser.add_argument("--print-ids", dest="pids", default=False,
                        action='store_true', help="")

    parser.add_argument("--regression", "--reg", dest="reg", default=False,
                        action='store_true', help="")

    parser.add_argument("--MoT", "-mot", dest="mot", default=False,
                        action='store_true', help="")
    parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                        help="The path to the word embeddings file (Word2Vec format)")
    parser.add_argument("--use-oracle-justifaication", "-oracle", dest="oracle", default=False,
                        action='store_true', help="use gold justification at inference")
    parser.add_argument("--regularize_vectors", "-regularize", dest="regularize", default=False,
                        action='store_true', help="regularize vector norms to 1.")
    parser.add_argument("--print-mapping", "-map", dest="map", default=False,
                        action='store_true', help="output scatter plot of hidden states")

    parser.add_argument("--save-final_sates", "-sf", dest="save_fin", default=False,
                        action='store_true', help="Save result of final epoch")

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
    parser.add_argument("--soft-triple-loss", "-stl", dest="stl", default=False,
                        action='store_true', help="")
    parser.add_argument("--cosine-triplet-loss", "-ctl", dest="ctl", default=False,
                        action='store_true', help="learning by cosine triplet loss")
    parser.add_argument("--dot-triplet-loss", "-dtl", dest="dtl", default=False,
                        action='store_true', help="learning by dot triplet loss")
    parser.add_argument("-m", "--margin", dest="margin", type=float, metavar='<float>',
                        default=0.2, help="satt loss ratio")
    parser.add_argument("--output-trustscore", dest="output_trustscore", default=False,
                        action='store_true', help="")
    parser.add_argument("-num", "--data_num", dest="num", type=int, metavar='<int>',
                        default=0, help="data number of used data for management")
    parser.add_argument("-seed", dest="seed", type=int,
                        metavar='<int>', default=-1)
    parser.add_argument("--cosface", "-cf", action='store_true',
                        default=False, help="use CosFace")
    parser.add_argument("--arcface", "-af", action='store_true',
                        default=False, help="use ArcFace")
    parser.add_argument("--sphereface", "-spf", action='store_true',
                        default=False, help="use SphereFace")
    parser.add_argument('--triplet_n_neighbors', type=int, default=None,
                        help='n_neighbors for triplet loss')
    parser.add_argument('--classter_num', type=int, default=1,
                        help='number of classter per class (for soft-triple-loss)')
    parser.add_argument('--inference', type=str, default="knn",
                        help='decode methods for inference')
    parser.add_argument("--map_embedding_per_epoch", action='store_true',
                        default=False, help="save mapping per epoch")
    parser.add_argument("--outdim", type=int, metavar='<int>',
                        default=300, help="output dimension.")
    parser.add_argument("--save_token_norm", "-stn", action='store_true',
                        default=False, help="save token norm per epoch by jsonl")
    parser.add_argument("--beta", dest="beta", type=float, metavar='<float>',
                        default=None, help="precision and recall ratio")
    parser.add_argument("--include_norm_for_att", "-infa", dest="include_norm_for_att", default=False,
                        action='store_true', help="By using this, the norm of the token is taken into account in the attention")
    parser.add_argument("--attention_train_size", type=int, metavar='<int>',
                        default=None, help="train data size for attentino")
    parser.add_argument("--make_feature_map", dest="make_feature_map", default=False,
                        action='store_true', help="")
    parser.add_argument("--measure_justification_identification", dest="measure_justification_identification", default=False,
                        action='store_true', help="")
    parser.add_argument("--measure_faithfulness_eraser", dest="measure_faithfulness_eraser", default=False,
                        action='store_true', help="")
    parser.add_argument("--correct_miss_separate", "-cms", dest="correct_miss_separate", default=False,
                        action='store_true', help="根拠箇所推定の評価の時に得点予測を間違ったものとあってたもので分けて評価する")
    parser.add_argument("--flip_mode", dest="flip_mode", default=False,
                        action='store_true', help="削除率の計算方法を「得点が変化するまでの削除率」にする")
    parser.add_argument("--accuracy_mode", dest="accuracy_mode", default=False,
                        action='store_true', help="性能をaccuracyで測る")
    # parser.add_argument("--update_bert_params", default=False,
    #                     action='store_true', help="update bert paramerters")
    parser.add_argument("--update_bert_params", default=0,
                        type=int, help="update bert paramerters")
    parser.add_argument("--config", type=str, help="bert config", choices=[
                        "cl-tohoku/bert-base-japanese-char-whole-word-masking", "cl-tohoku/bert-base-japanese-whole-word-masking", "bert-base-uncased"])
    parser.add_argument("--implementation", type=str, default="funayama",
                        choices=["funayama", "asazuma", "is_attention_interpretable", "no_lstm_bert"])
    parser.add_argument("--no_use_fuka_ann",
                        action="store_true", default=False)
    args = parser.parse_args()

    return args


def train(args):
    # from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    from transformers import BertConfig
    import getinfo
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    # 使わないけどエラーが出るから書いておく
    args.metric = "triplet"
    info = getinfo.TrainInfo(args)
    info.store_part_scoring_info()

    info.store_metric_info(args)
    info.store_item_info(args.item)
    if args.npt is not None:
        import neptune
        neptune_init(
            neptune=neptune, project_name=args.npt[0], experiment_name=args.npt[1], info=info)
        neptune.set_property("Data number", args.num)

    # L.set_logger(out_dir=info.out_dir, debug=args.debug)
    L.set_logger(out_dir=info.out_dir, debug=args.debug)
    logger = getLogger(__name__)
    logger.info(f"{sys.argv}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # from model import BertForPartScoringWithLSTM, BiRnnModel, BiRnnModelForItemScoringWithMetric
    from model_eraser import BiRnnModelForItemScoringERASER, BertFinetuningForItemScoringERASER
    from model_asazuma import LSTMAttention

    import part_scoring_info as psi

    logger.info(f"Prompt name :{info.prompt_name}")
    if args.BERT:
        from sas.input_data import input_data_ps_BERT as input_data
    else:
        from sas.input_data import input_data_ps as input_data

    (train_x, train_y, train_mask, train_attention, attention_flag), (dev_x, dev_y,
                                                                      dev_mask, dev_attention), (test_x, test_y, test_mask, test_attention) = input_data(info)

    item_num = util.get_item_num(args.item, info.main_factors)
    info.item_max_score = info.ps_labels[item_num][1]

    info.device = device
    info.train_size = train_x.shape[0]
    if args.npt is not None:
        neptune.set_property("train size", info.train_size)
    logger.debug(f"factor name :{info.factors}")
    logger.debug(f"ps_labels:{info.ps_labels}")
    logger.info(f"Scoring Item {args.item}")

    # if args.BERT:
    #     if args.char:
    #         config = BertConfig.from_pretrained(
    #             "cl-tohoku/bert-base-japanese-char-whole-word-masking")
    #     else:
    #         config = BertConfig.from_pretrained(
    #             "cl-tohoku/bert-base-japanese-whole-word-masking")
    #     # embeddingの次元数を調整
    #     info.emb_dim = 768
    #     model = BiRnnModelForItemScoringWithMetric(
    #         info, config=config)
    #     model.freeze_bert_pram()
    # else:
    #     model = BiRnnModelForItemScoringWithMetric(
    #         info)

    # if args.model_path != None:
    #     model.load_state_dict(torch.load(args.model_path))
    #     logger.info(f"Load pretrained model from {args.model_path}")

    from metric_learning.triplet_loss import OnlineTripletLoss, ImprovedOnlineTripletLoss, TrustScoreOnlineTripletLoss, NomalizedRankedTripletLoss, NormalizedDualOnlineTripletLoss
    from metric_learning.metrics import ArcMarginProduct, AddMarginProduct, SphereProduct
    from metric_learning.softtriple_loss import SoftTripleLoss
    from metric_learning.angle_base_loss import CosineTripletLoss, DotTripletLoss
    if args.arcface:
        logger.info(f"Loss: ArcFace")
        metric = "ArcFace"
        # face_model = ArcMarginProduct(
        #     info.rnn_dim, info.item_max_score + 1).to(device)
        face_model = ArcMarginProduct(
            args.outdim, info.item_max_score + 1).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.cosface:
        logger.info(f"Loss: CosFace")
        metric = "CosFace"
        # face_model = AddMarginProduct(
        #     info.rnn_dim, info.item_max_score + 1).to(device)
        face_model = AddMarginProduct(
            args.outdim, info.item_max_score + 1).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.sphereface:
        logger.info(f"Loss: SphereFace")
        metric = "SphereFace"
        # face_model = SphereProduct(
        #     info.rnn_dim, info.item_max_score + 1).to(device)
        face_model = SphereProduct(
            args.outdim, info.item_max_score + 1).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    elif args.stl:
        logger.info(f"Loss: SoftTriple Loss")
        # soft_triple_loss = SoftTripleLoss(
        #     info.rnn_dim, info.item_max_score + 1, num_initial_center=args.classter_num).to(device)
        soft_triple_loss = SoftTripleLoss(
            args.outdim, info.item_max_score + 1, num_initial_center=args.classter_num).to(device)
        metric = "SoftTriple Loss"
    elif args.ctl:
        logger.info(f"Loss: Cosine triplet loss")
        triplet_loss = CosineTripletLoss()
        metric = "Cosine Triplet Loss"
    elif args.dtl:
        logger.info(f"Loss: Dot triplet loss")
        triplet_loss = DotTripletLoss()
        metric = "Dot Triplet Loss"
    elif args.imp:
        logger.info(f"Loss: Improved triplet loss")
        triplet_loss = ImprovedOnlineTripletLoss(args.margin, 0.01)
        metric = "Improved Triplet Loss"
    elif args.tsl:
        logger.info(f"Loss: TrustScore Triplet Loss")
        triplet_loss = TrustScoreOnlineTripletLoss(0.8)
        metric = "TrustScore Triplet Loss"
    elif args.rtln:
        logger.info(f"Loss: Normalized Ranked Triplet Loss")
        triplet_loss = NomalizedRankedTripletLoss(args.margin, p=1)
        metric = "Ranked Triplet Loss"
    elif args.qrtln:
        logger.info(f"Loss: Normalized Quadratic Ranked Triplet Loss")
        triplet_loss = NomalizedRankedTripletLoss(args.margin, p=2)
        metric = "Quadratic Ranked Triplet Loss"
    elif args.drln:
        logger.info(f"Loss: Normalized Dual Ranked Triplet Loss")
        triplet_loss = NormalizedDualOnlineTripletLoss(args.margin, p=1)
        metric = "Dual Ranked Triplet Loss"
    else:
        if args.triplet_n_neighbors is None:
            logger.info(f"Loss: Triplet Loss")
            triplet_loss = OnlineTripletLoss(args.margin)
        else:
            logger.info(
                f"Loss: Triplet Loss {args.triplet_n_neighbors}nn")
            triplet_loss = OnlineTripletLoss(
                args.margin, args.triplet_n_neighbors)
        metric = "Triplet Loss"

    info.metric = metric
    if args.BERT:
        if args.char and "char" not in args.config:
            raise ValueError(f"{args.config} is not char model")
        elif not args.char and "char" in args.config:
            raise ValueError(f"{args.config} is char model")
        config = BertConfig.from_pretrained(args.config)
        info.emb_dim = config.hidden_size
        if info.implementation == 'no_lstm_bert':
            model = BertFinetuningForItemScoringERASER(
                info, config=args.config)
        elif info.implementation == 'funayama':
            model = BiRnnModelForItemScoringERASER(info, config=args.config)
        else:
            raise ValueError(f"we don't have {info.implementation} model")
        # if args.update_bert_params:
        #     model.freeze_bert_pram()
        #     model.unfreeze_bert_layer_param(-1)
        #     model.unfreeze_bert_layer_param(-2)
        # else:
        #     model.freeze_bert_pram()
        model.freeze_bert_pram()
        for i in range(1, args.update_bert_params + 1):
            model.unfreeze_bert_layer_param(-1 * i)
    else:
        if info.implementation == 'asazuma':
            model = LSTMAttention(info)
        elif info.implementation == 'funayama':
            model = BiRnnModelForItemScoringERASER(info)
        else:
            raise ValueError(f"we don't have {info.implementation} model")

    model.to(device)
    from sas.optimizer import get_optimizer
    optimizer = get_optimizer(info.opt, model)

    logger.info(f"Optimizer: {optimizer}")
    if args.npt is not None:
        neptune.set_property("Optimizer", optimizer)
        neptune.set_property("Metric", metric)

    from Dataset import part_scoring_set

    dataset = part_scoring_set(
        train_x, train_y, train_mask, train_attention, attention_flag)

    dataloader = data.DataLoader(
        dataset, **{'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0})

    eval_dataloader = data.DataLoader(
        dataset, **{'batch_size': 400, 'shuffle': False, 'num_workers': 0})

    # 初期化
    best = -1.0
    best_mses = (1.0, 1.0)

    test_best = -1
    total_train_time = 0

    import evaluator_metric

    logger.debug(f"train shape:{train_x.shape}")
    logger.debug(f"dev shape: {dev_x.shape}")
    logger.debug(f"test shape:{test_x.shape}")
    train_data_size = train_x.shape[0]
    eval = evaluator_metric.EvaluatorForItemScoring(
        info, dev_x.to(device), test_x.to(device), dev_y, test_y, dev_mask.to(device), test_mask.to(device), dev_attention[item_num].to(device), test_attention[item_num].to(device), item_num, affin=True, normalize=(args.sphereface or args.cosface or args.arcface or args.ctl or args.dtl), criterion=triplet_loss)

    # グラフ描画用のdf
    # df = pd.DataFrame(
    #     columns=['sentence', 'score', 'marker_size', 'epoch', "X", "Y", "Z"])
    # from sas.input_data import get_answers
    # answers = get_answers(info)

    # ノルム保存用のインスタンス
    # norm_dict = Norm()
    # token_dict = TokenDict()

    # tokenごとのノルム保存
    # from sas.input_data import get_justification_word_set
    # justification_word_set = get_justification_word_set(info)

    for ii in range(args.epochs):
        print(ii, info.metric, metric)
        b_loss = list()
        t0 = time()
        model.train()
        #
        # train_hidden_states = [torch.zeros(
        #     train_shape[0], 2) for k in info.ps_labels]
        # train_golds = [torch.zeros(
        #     train_shape[0]) for k in info.ps_labels]
        index = 0
        batch_att_loss = torch.tensor(0.0).to(device)
        batch_scoring_loss = torch.tensor(0.0).to(device)
        i = item_num

        for num, (batch_x, batch_y, batch_m, batch_a, batch_f) in enumerate(dataloader):
            mini_batch_size = batch_y[0].shape[0]
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_a = batch_a[item_num].to(device)

            outputs = model(batch_x, token_type_ids=None,
                            attention_mask=batch_m)

            output = outputs[0]
            attn_output = outputs[1]
            hidden_states = outputs[2][0]

            if args.include_norm_for_att:
                att_norm = torch.norm(outputs[2][2], dim=-1)
                assert att_norm.size() == attn_output.size(
                ), f"{att_norm.size()}, {attn_output.size()}"
                att_norm = att_norm * attn_output / \
                    torch.unsqueeze(torch.norm(att_norm, dim=-1), dim=-1)
                # att_norm = F.softmax(att_norm, dim=-1)

            else:
                att_norm = attn_output

            # if args.affin:
            #     target_embeddings = output
            # else:
            #     target_embeddings = hidden_states
            target_embeddings = output

            loss = torch.tensor(0.0).to(device)
            loss_satt = torch.tensor(0.0).to(device)
            if metric.endswith("Triplet Loss"):
                t_loss = torch.tensor(0.0).to(device)

                triplet_num = 0
                loss_flag = False
                batch_y_ps = batch_y[i +
                                     1].type(torch.LongTensor).abs().to(device)
                batch_f = batch_f[i + 1].to(device)
                # train_hidden_states[i][index:index + batch_x.shape[0], :] = target_embeddings
                # train_golds[i][index:index + batch_x.shape[0]] = batch_y[i + 1]
                # tripletが作れるとき
                if len(torch.unique(batch_y_ps)) != 1:
                    triplet_l = triplet_loss(
                        target_embeddings, batch_y_ps, info.ps_labels[i][1])
                    if triplet_l[1] != 0:
                        t_loss += triplet_l[0]
                        triplet_num += triplet_l[1]
                        loss_flag = True

                if i < len(info.main_factors) and info.satt:
                    p_len = attn_output.shape[1]
                    if args.beta is None:
                        loss_satt += attn_loss(att_norm[batch_f],
                                               batch_a[:, :p_len][batch_f])
                        # print(loss_satt, att_norm[batch_f],
                        #       batch_a[:, :p_len][batch_f])
                    else:
                        loss_satt += modified_attn_loss(
                            att_norm[batch_f], batch_a[:, :p_len][batch_f], args.beta)
                    loss_flag = True

                loss += t_loss

                if info.satt:
                    loss += loss_satt * args.lamda
                    batch_att_loss += loss_satt * batch_x.shape[0]
                if loss_flag:
                    batch_scoring_loss += t_loss * batch_x.shape[0]
                    b_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                index += batch_x.shape[0]

            elif metric == ("SoftTriple Loss"):
                # softtriple_lossの時
                # cosfaceなど
                batch_y_ps = batch_y[i +
                                     1].type(torch.LongTensor).abs().to(device)
                f_loss = soft_triple_loss(target_embeddings, batch_y_ps)
                loss += f_loss

                if i < len(info.main_factors) and info.satt:
                    p_len = attn_output.shape[1]
                    loss_satt += attn_loss(attn_output, batch_a[:, :p_len])
                if info.satt:
                    loss += loss_satt * args.lamda
                    batch_att_loss += loss_satt * batch_x.shape[0]

                batch_scoring_loss += f_loss * batch_x.shape[0]
                b_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                index += batch_x.shape[0]

            else:
                # cosfaceなど
                batch_y_ps = batch_y[i +
                                     1].type(torch.LongTensor).abs().to(device)
                outputs = face_model(target_embeddings, batch_y_ps)
                f_loss = criterion(outputs, batch_y_ps)
                loss += f_loss

                if i < len(info.main_factors) and info.satt:
                    p_len = attn_output.shape[1]
                    loss_satt += attn_loss(attn_output, batch_a[:, :p_len])
                if info.satt:
                    loss += loss_satt * args.lamda
                    batch_att_loss += loss_satt * batch_x.shape[0]

                batch_scoring_loss += f_loss * batch_x.shape[0]
                b_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                index += batch_x.shape[0]

        tr_time = time() - t0
        total_train_time += tr_time
        if metric.endswith("Triplet Loss"):
            logger.info(
                f"Epoch:{ii} Train time :{tr_time:.2f} (s) Number of total triplets: {triplet_num}")
        else:
            logger.info(
                f"Epoch:{ii} Train time :{tr_time:.2f} (s)")
        instance_scoring_loss = batch_scoring_loss.item() / train_data_size
        instance_att_loss = batch_att_loss.item() / train_data_size
        logger.info(f"Scoring loss: {instance_scoring_loss:.3f}")
        logger.info(f"Attention loss: {instance_att_loss:.3f}")

        (dev_qwks, test_qwks, dev_mses, test_mses), (train_embeddings, train_golds, train_attention), (dev_outputs, test_outputs), (d_pred,
                                                                                                                                    t_pred), (dev_target, test_target), (d_dist, t_dist), (d_ev, t_ev), (dev_grads, test_grads) = eval.evaluate(info, model, eval_dataloader)
        # ここでtrain_dataの学習の様子を描画するデータをappend
        # devのhidden_stateを描画する用
        model.eval()
        with torch.no_grad():
            # if args.affin:
            #     dev_hidden_states = dev_outputs[0]
            #     test_hidden_states = test_outputs[0]
            # else:
            #     dev_hidden_states = dev_outputs[2][0]
            #     test_hidden_states = test_outputs[2][0]
            dev_hidden_states = dev_outputs[0]
            test_hidden_states = test_outputs[0]

            # normの情報をappend
            # norm_dict.append(train_embeddings.cpu(
            # ).detach().numpy(), 'train')
            # norm_dict.append(
            #     dev_hidden_states.cpu().detach().numpy(), 'dev')
            # norm_dict.append(test_hidden_states.cpu(
            # ).detach().numpy(), 'test')

            # epochごとの答案ベクトルを保存
            # if args.map_embedding_per_epoch is True:
            #     # 可視化する時のコード
            #     assert args.outdim == 2
            #     # train dataを追加
            #     df_train = pd.DataFrame(data={"id": [f"train_{i}" for i in range(train_golds.size(0))], "sentence": answers['train'], "score": list(map(lambda x: f"gold_{x}", train_golds.cpu().detach().numpy().tolist())), "epoch": [ii for _ in range(train_golds.size(0))], "X": train_embeddings[:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": train_embeddings[:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     # dev data 追加
            #     df_dev = pd.DataFrame(data={"id": [f"dev_{i}" for i in range(len(d_pred))], "sentence": answers['dev'], "score": list(map(lambda x: f"pred_{x}", dev_target[item_num+1])), "epoch": [ii for _ in range(len(d_pred))], "X": dev_hidden_states[:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": dev_hidden_states[:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     # test data 追加
            #     df_test = pd.DataFrame(data={"id": [f"test_{i}" for i in range(len(t_pred))], "sentence": answers['test'], "score": list(map(lambda x: f"test_{x}", test_target[item_num+1])), "epoch": [ii for _ in range(len(t_pred))], "X": test_hidden_states[:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": test_hidden_states[:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     df = df.append(df_train)
            #     df = df.append(df_dev)
            #     df = df.append(df_test)

            # # 単語のベクトルを追加
            # df_token = pd.DataFrame(data={"id": [f"token_{i}" for i in range(len(t_pred))], "sentence": , "score": list(map(lambda x: f"test_{x}", test_target[item_num+1])), "epoch": [ii for _ in range(len(t_pred))], "X": test_hidden_states[:, 0].cpu().detach().numpy().reshape(-1).tolist(), "Y": test_hidden_states[:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            # df = df.append(df_token)

            # tokenごとのノルムを追加
            # if args.save_token_norm or args.map_embedding_per_epoch:
            #     token_list = []
            #     id_list = []
            #     X = []
            #     Y = []
            #     assert len(answers['dev']) == dev_outputs[2][2].size(0)
            #     for sentence, embeddings in zip(answers['dev'], dev_outputs[2][2]):
            #         tokens = sentence.split(" ")
            #         for token, embedding in zip(tokens, embeddings):
            #             if token in justification_word_set:
            #                 emb = embedding.cpu().detach().numpy()
            #                 if args.save_token_norm:
            #                     # 同じepochに同じtokenが合った場合はとりあえず上書きしておく
            #                     token_dict.token_dict[token][ii] = float(
            #                         np.linalg.norm(emb))
            #                 if args.map_embedding_per_epoch:
            #                     if token not in token_list:
            #                         token_list.append(token)
            #                         id = info.vocab.get(token, 1)
            #                         output = model(torch.LongTensor(
            #                             [[id]]).to(device), token_type_ids=None, attention_mask=torch.LongTensor([[1]]).to(device))

            #                         attention = output[1][0][0].item()
            #                         assert attention == 1, f"{attention}"
            #                         token_emb = output[0][0].detach(
            #                         ).cpu().numpy().tolist()
            #                         X.append(token_emb[0])
            #                         Y.append(token_emb[1])

            # tokenもグラフに可視化する
            # if args.map_embedding_per_epoch:
            #     df_token = pd.DataFrame(data={"id": token_list, "sentence": token_list, "score": [-1 for _ in range(
            #         len(token_list))], "epoch": [ii for _ in range(len(token_list))], "X": X, "Y": Y})
            #     df = df.append(df_token)

            # logger_eval.info(f"[Dev]: {dev_qwk:.3f} [Test]: {test_qwk:.3f}")
            if best <= dev_qwks:
                best = dev_qwks
                best_mses = dev_mses
                test_best = test_qwks
                test_best_mses = test_mses
                b_epoch = ii
                best_test_outputs = test_outputs
                best_dev_outputs = dev_outputs
                best_dev_pred = d_pred
                best_test_pred = t_pred
                best_d_dist = d_dist
                best_t_dist = t_dist
                best_d_ev = d_ev
                best_t_ev = t_ev
                best_dev_attentions = best_dev_outputs[1]
                best_test_attentions = best_test_outputs[1]
                # best_dev_norms = torch.norm(best_dev_outputs[2][2], dim=-1)
                # best_test_norms = torch.norm(best_test_outputs[2][2], dim=-1)
                best_dev_norms = None
                best_test_norms = None

                # best_dev_grads_norm = torch.norm(dev_grads, dim=-1)
                # best_test_grads_norm = torch.norm(test_grads, dim=-1)
                best_dev_grads_norm = None
                best_test_grads_norm = None

                handling_data.save_hidden_states_metric_item(
                    info, train_embeddings, attn=train_attention, target=train_golds, prefix="train")
                handling_data.save_hidden_states_metric_item(
                    info, dev_outputs, pred=d_pred, attn=best_dev_attentions, norm=best_dev_norms, grad=best_dev_grads_norm, target=dev_target[item_num + 1], dist=d_dist, ev=best_d_ev, prefix="dev")
                handling_data.save_hidden_states_metric_item(
                    info, test_outputs, pred=t_pred, attn=best_test_attentions, norm=best_test_norms, grad=best_test_grads_norm, target=test_target[item_num + 1], dist=t_dist, ev=best_t_ev, prefix="test")
                if not args.not_save_weights:
                    torch.save({
                        'epoch': ii,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, info.out_dir + "_best_checkpoint.pt")

                # ノルム*attentionの値を計算
                # best_dev_att_norm = torch.norm(
                #     best_dev_outputs[2][2], dim=-1)
                # best_dev_att_norm = best_dev_att_norm * best_dev_attentions / \
                #     torch.unsqueeze(torch.norm(
                #         best_dev_att_norm, dim=-1), dim=-1)
                # best_dev_att_norm = F.softmax(
                #     best_dev_att_norm * best_dev_attentions, dim=-1)
                # best_test_att_norm = torch.norm(
                #     best_test_outputs[2][2], dim=-1)
                # best_test_att_norm = best_test_att_norm * best_test_attentions / \
                #     torch.unsqueeze(torch.norm(
                #         best_test_att_norm, dim=-1), dim=-1)

                # 答案と単語のベクトルを2次元に圧縮して可視化したグラフを作成
                # from util import function
                # mapping_vec = []
                # sentence = []
                # mapping_vec.extend(train_embeddings.cpu().detach().numpy().tolist()
                # mapping_vec.extend(dev_hidden_states.cpu().detach().numpy().tolist()
                # mapping_vec.extend(test_hidden_states.cpu().detach().numpy().tolist()
                # sentence.extend(answers['train'])
                # sentence.extend(answers['dev'])
                # sentence.extend(answers['test'])
                # mapping_vec = function.dimention_reduction_umap(, metric=info.reduction_metric)

                # 単語のベクトルを保存
                # token_emb_of_best_epoch = defaultdict(list)
                # for sentence, embeddings, attentions in zip(answers['dev'], dev_outputs[2][2].cpu().detach().numpy().tolist(), dev_outputs[1].cpu().detach().numpy().tolist()):
                #     tokens = sentence.split(" ")
                #     for token, embedding, attention in zip(tokens, embeddings, attentions):
                #         token_emb_of_best_epoch[token].append(
                #             list(map(lambda x: x*attention, embedding)))
                # for token in token_emb_of_best_epoch.keys():
                #     token_emb_of_best_epoch[token] = np.mean(
                #         np.array(token_emb_of_best_epoch[token]), axis=0).tolist()

                # with open(f"{info.out_dir}_token_embedding_result.pickle", 'wb') as f:
                #     cloudpickle.dump(token_emb_of_best_epoch, f)

            if args.npt is not None:
                log_with_neptune(neptune=neptune, epoch=ii, dev_qwk=dev_qwks,
                                 scoring_loss=instance_scoring_loss, attention_loss=instance_att_loss)

    # ノルムを保存
    # norm_dict.save_norm(f"{info.out_dir}answer_norm_per_epoch.json")
    # if args.save_token_norm:
    #     token_dict.save_norm(f"{info.out_dir}_token_norm_per_epoch.json")

    if args.save_fin:
        handling_data.save_hidden_states_metric_item(
            info, train_embeddings, target=train_golds, prefix="train_final")
        handling_data.save_hidden_states_metric_item(
            info, dev_outputs, pred=d_pred, target=dev_target[item_num + 1], dist=d_dist, ev=d_ev, prefix="dev_final")
        # handling_data.save_hidden_states_metric_item(
        #     info, test_outputs,pred= t_pred,target= test_target[item_num+1], dist=t_dist, ev=t_ev, prefix="test_final")

    logger.info(f"Best epoch {b_epoch}")
    eval.print_info(best, best_mses, item_num, type="Dev", log=logger)
    eval.print_info(test_best, test_best_mses,
                    item_num, type="Test", log=logger)
    if args.npt is not None:
        neptune.log_metric('Test QWK', b_epoch, test_best)

    # 学習過程のmappingを可視化する
    if args.map_embedding_per_epoch is True:
        from util.mapping.map_embedding_per_epoch import save_map_embedding_per_epoch
        save_map_embedding_per_epoch(
            df, f"{info.out_dir}_embedding_per_epoch_{args.outdim}dim.html")

# logger.info(
    #    f"BEST -> [Dev]: {best:.3f} [Test]: {test_best:.3f} ({b_epoch} epoch)")
    handling_data.save_result_part_scoring_metric_item_scoring(
        info, best_test_pred, test_mask.sum(1), data_path=info.test_dir, item_num=item_num, probs=True, norms=None, attentions=best_test_outputs[1], distance=best_t_dist, evidence=best_t_ev, prefix="test")
    handling_data.save_result_part_scoring_metric_item_scoring(
        info, best_dev_pred, dev_mask.sum(1), data_path=info.dev_dir, item_num=item_num, probs=True, norms=None, attentions=best_dev_outputs[1], distance=best_d_dist, evidence=best_d_ev, prefix="dev")

    with open(info.out_dir + '_train_info.pickle', mode='wb') as f:
        cloudpickle.dump(info, f)

    if args.map:
        import util.mapping.map_embedding as map_embedding

        fig_best = map_embedding.map_embedding(info.out_dir + "_train_result.pickle", info.out_dir +
                                               "_dev_result.pickle", info.train_dir, info.dev_dir, args.item, info.out_dir + "_scatter_best.html")
        if args.save_fin:
            fig_final = map_embedding.map_embedding(info.out_dir + "_train_final_result.pickle", info.out_dir +
                                                    "_dev_final_result.pickle", info.train_dir, info.dev_dir, args.item, info.out_dir + "_scatter_final.html")
        if args.npt is not None:
            from neptunecontrib.api import log_chart
            log_chart(
                name=f'{info.prompt_name}_{info.item}_e{b_epoch}_scatter_best', chart=fig_best)
            if args.save_fin:
                log_chart(
                    name=f'{info.prompt_name}_{info.item}_scatter_final', chart=fig_final)

    if args.output_trustscore:
        from confidence_estimation.trust_score import output_trustscore
        logger.info("Calculate confidence score...")
        output_trustscore(train_pk_dir=info.out_dir + "_train_result.pickle", target_pk_dir=info.out_dir + "_dev_result.pickle",
                          target_dir=info.dev_dir, save=info.out_dir + "_confidence_dev.xlsx", max_score=info.item_max_score)
        output_trustscore(info.out_dir + "_train_result.pickle", info.out_dir + "_test_result.pickle",
                          info.test_dir, info.out_dir + "_confidence_test.xlsx", info.item_max_score)
        logger.info("Done.")

    if args.make_feature_map:
        # best modelをロードする
        from analysis.util import load_instance_based_model, get_explanation_pickle_data
        model = load_instance_based_model(info)
        # yだけ全体点の情報も入ってるので修正
        train_y = train_y[item_num + 1]
        dev_y = dev_y[item_num + 1]
        test_y = test_y[item_num + 1]

        # 特徴マップを作成
        from make_feature_map import make_explanation_file_for_instancebase
        from interpretability.explanation import Explanation
        explanation = Explanation(model, info)

        make_explanation_file_for_instancebase(
            info, explanation, dev_x, dev_y, dev_mask, "dev")
        make_explanation_file_for_instancebase(
            info, explanation, test_x, test_y, test_mask, "test")

        # explanationデータをロード
        dev_explanation_df = get_explanation_pickle_data(info, "dev")
        test_explanation_df = get_explanation_pickle_data(info, "test")

        if args.measure_justification_identification:
            # justification_identificationをする
            from measure_justification_identification import make_justification_identification_file
            from analysis.util import get_sas_list
            # 訓練データと予測データのjsonファイルを取得
            _, dev_gold_justification_list, dev_gold_score_list, _, _ = get_sas_list(
                info, "dev")
            _, test_gold_justification_list, test_gold_score_list, test_pred_score_list, _ = get_sas_list(
                info, "test")
            # 0点を含めるか否かはdmyトークンを使用しているかで決定
            make_justification_identification_file(
                info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, args.dmy)

            # if args.measure_faithfulness_eraser:
            #     # eraserでfaithfulnessを計算
            #     from measure_faithfulness_eraser import make_faithfulness_eraser_file
            #     make_faithfulness_eraser_file(info,  test_pred_score_list, test_explanation_df,
            #                                   test_x, test_y, test_mask, test_attention, model)

    return info, b_epoch, best, best_mses, test_best, test_best_mses


def main():
    args = parse_args()
    info, b_epoch, best, best_mses, test_best, test_best_mses = train(args)
    handling_data.save_res(info, b_epoch, best, best_mses,
                           test_best, test_best_mses)


def log_with_neptune(neptune, epoch, dev_qwk, scoring_loss, attention_loss):
    neptune.log_metric('Dev QWK', epoch, dev_qwk)
    neptune.log_metric('Scoring_loss', epoch, scoring_loss)
    neptune.log_metric('Attention_loss', epoch, attention_loss)


def log_loss_function_with_neptune(neptune, metric):
    neptune.set_property("Metric", metric)


def neptune_init(neptune, project_name, experiment_name, info):
    neptune.init(project_qualified_name=project_name)
    neptune.create_experiment(name=experiment_name, params=info.__dict__)


if __name__ == '__main__':
    main()
