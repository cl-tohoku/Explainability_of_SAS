from torch.utils import data
import torch
import numpy as np
import torch.nn
from sas.quadratic_weighted_kappa import quadratic_weighted_kappa as qwk
from logging import setLoggerClass, getLogger
from util import logger as L
import sys
from time import time
from loss import attn_loss, cross_entropy_with_penalty
import copy
import argparse
from sas import handling_data, util
from os import path
import pandas as pd
import plotly.express as px
from util.save_norm import Norm
from pytorch_memlab import MemReporter
from pytorch_memlab import profile


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

    parser.add_argument("--mse-based", dest="mseb", default=False,
                        action='store_true', help="")
    parser.add_argument("-satt", dest="satt", default=False,
                        action='store_true', help="")
    parser.add_argument("-num", "--data_num", dest="num", type=int, metavar='<int>',
                        default=0, help="data number of used data for management")
    parser.add_argument("--BERT", "-BERT", dest="BERT", default=False,
                        action='store_true', help="")
    parser.add_argument("--neputune", "-npt", dest="npt", type=str,
                        help="run with neptune.ai. ex. -npt user_name/project_name experiment_name", nargs=2)

    parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>',
                        default=300, help="RNN dimension.")
    parser.add_argument("-emb_dim", "--emb_dim", dest="emb_dim", type=int, metavar='<int>',
                        default=100, help="RNN dimension.")
    parser.add_argument("-v_size", "--vocab_size", dest="vocab_size", type=int, metavar='<int>',
                        default=4000, help="upper bound of vocab size")
    parser.add_argument("-l", "--lamda", dest="lamda", type=float, metavar='<float>',
                        default=1.0, help="satt loss ratio")
    parser.add_argument("--optimizer", "-opt", dest="opt", default='rmsprop',
                        choices=['rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam'], help="")

    parser.add_argument("--save-final_sates", "-sf", dest="save_fin", default=False,
                        action='store_true', help="Save result of final epoch")
    parser.add_argument("--output-trustscore", dest="output_trustscore", default=False,
                        action='store_true', help="")
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

    parser.add_argument("--dummy", "-dmy", dest="dmy", default=False,
                        action='store_true', help="")

    parser.add_argument("--item", "-item", dest="item")

    parser.add_argument("--print-ids", dest="pids", default=False,
                        action='store_true', help="")
    parser.add_argument("--regression", "-reg", dest="reg", default=False,
                        action='store_true', help="")
    parser.add_argument("--update-embedding", "-ue", dest="update_embed", default=False,
                        action='store_true', help="")
    parser.add_argument("--MoT", "-mot", dest="mot", default=False,
                        action='store_true', help="")
    parser.add_argument("--penalty_loss", "-plt", dest="plt", default=False,
                        action='store_true', help="")
    parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>',
                        help="The path to the word embeddings file (Word2Vec format)")
    parser.add_argument("--print-mapping", "-map", dest="map", default=False,
                        action='store_true', help="output scatter plot of hidden states")
    parser.add_argument("-seed", dest="seed", type=int,
                        metavar='<int>', default=-1)
    parser.add_argument("--metric", "-metric", dest="metric", default="crossentropy",
                        type=str, help="crossentropy or cosface or arcface or sphereface")
    parser.add_argument("--map_embedding_per_epoch", action='store_true',
                        default=False, help="save mapping per epoch")
    parser.add_argument("--outdim", type=int, metavar='<int>',
                        default=300, help="output dimension.")
    parser.add_argument("--attention_train_size", type=int, metavar='<int>',
                        default=None, help="train data size for attentino")
    parser.add_argument("--make_feature_map",  dest="make_feature_map", default=False,
                        action='store_true', help="")
    parser.add_argument("--measure_justification_identification",  dest="measure_justification_identification", default=False,
                        action='store_true', help="")
    parser.add_argument("--measure_faithfulness_eraser",  dest="measure_faithfulness_eraser", default=False,
                        action='store_true', help="")
    parser.add_argument("--correct_miss_separate", "-cms", dest="correct_miss_separate", default=False,
                        action='store_true', help="根拠箇所推定の評価の時に得点予測を間違ったものとあってたもので分けて評価する")
    parser.add_argument("--flip_mode", dest="flip_mode", default=False,
                        action='store_true', help="削除率の計算方法を「得点が変化するまでの削除率」にする")
    parser.add_argument("--accuracy_mode", dest="accuracy_mode", default=False,
                        action='store_true', help="性能をaccuracyで測る")
    args = parser.parse_args()

    return args


def main():
    # from bert_scripts.tokenization import MecabBertTokenizer, MecabCharacterBertTokenizer
    from transformers import BertConfig
    import json
    import getinfo

    args = parse_args()
    if args.seed >= 0:
        torch.manual_seed(args.seed)
    info = getinfo.TrainInfo(args)
    info.store_part_scoring_info()
    info.store_item_info(args.item)
    if args.npt is not None:
        import neptune
        neptune_init(
            neptune=neptune, project_name=args.npt[0], experiment_name=args.npt[1], info=info)
        neptune.set_property("Data number", args.num)
        if args.plt:
            Metric = "CE with plt"
        else:
            Metric = "CE"
        neptune.set_property("Metric", Metric)

    L.set_logger(out_dir=info.out_dir, debug=args.debug)
    logger = getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info.store_device(device)
    from model import BertForScoring, BertForPartScoring, BertForPartScoringWithLSTM, BiRnnModel, BiRnnModelForItemScoring

    import part_scoring_info as psi
    logger.info(f"{sys.argv}")

    prompt = psi.prompt_check(info.train_dir)[0]
    logger.info(f"Prompt name :{prompt}")
    if args.BERT:
        from sas.input_data import input_data_ps_BERT as input_data
    else:
        from sas.input_data import input_data_ps as input_data

    (train_x, train_y, train_mask, train_attention, attention_flag), (dev_x, dev_y,
                                                                      dev_mask, dev_attention), (test_x, test_y, test_mask, test_attention) = input_data(info)

    '''
    answer, scores_ps, ps_labels, info.main_factors, info.ded_factors = input_data.read_data_ps(
        info.test_dir, prompt)
    '''

    info.train_size = train_x.shape[0]
    if args.npt is not None:
        neptune.set_property("train size", info.train_size)
        neptune.set_property("Penalty", args.plt)

    logger.debug(f"factor name :{info.factors}")
    logger.debug(f"ps_labels:{info.ps_labels}")
    item_num = util.get_item_num(args.item, info.main_factors)
    info.item_max_score = info.ps_labels[item_num][1]
    logger.info(f"Scoring Item {args.item}")

    if args.BERT:
        if args.BERT:
            if args.char:
                config = BertConfig.from_pretrained(
                    "cl-tohoku/bert-base-japanese-char-whole-word-masking")
            else:
                config = BertConfig.from_pretrained(
                    "cl-tohoku/bert-base-japanese-whole-word-masking")
        info.emb_dim = config.hidden_size
        model = BiRnnModelForItemScoring(info, config=config)
        model.freeze_bert_pram()
    else:
        model = BiRnnModelForItemScoring(info)

    model.print_model_info()

    # reporter = MemReporter(model)
    # # ③ 訓練前にレポートすることで、モデルのアーキテクチャが使っているメモリがわかる。
    # reporter.report()

    if args.model_path != None:
        model.load_state_dict(torch.load(args.model_path))
        logger.info(f"Load pretrained model from {args.model_path}")
    # model.unfreeze_bert_layer_param(-1)

    model.to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # criterion_satt = torch.nn.MSELoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters())
    from sas.optimizer import get_optimizer
    optimizer = get_optimizer(info.opt, model)

    logger.info(f"Optimizer: {optimizer}")

    from Dataset import part_scoring_set
    dataset = part_scoring_set(
        train_x, train_y, train_mask, train_attention, attention_flag)

    dataloader = data.DataLoader(
        dataset, **{'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0})

    eval_dataloader = data.DataLoader(
        dataset, **{'batch_size': 64, 'shuffle': False, 'num_workers': 0})

    best = -1.0
    best_mses = (1.0, 1.0)

    dev_best = 1.0
    test_best = 1
    total_train_time = 0

    logger_train = getLogger('Training')
    logger_eval = getLogger('Evaluation')
    import evaluator_cls
    import evaluator_reg

    logger.debug(f"train shape:{train_x.shape}")
    logger.debug(f"dev shape: {dev_x.shape}")
    logger.debug(f"test shape:{test_x.shape}")

    if args.reg:
        criterion = torch.nn.MSELoss()
    else:
        if args.plt:
            criterion = cross_entropy_with_penalty
        else:
            criterion = torch.nn.CrossEntropyLoss()

    if args.reg:
        eval = evaluator_reg.EvaluatorForItem(
            info, dev_x.to(device), test_x.to(device), dev_y, test_y, dev_mask.to(device), test_mask.to(device), item_num)
    else:
        eval = evaluator_cls.EvaluatorForItem(
            info, dev_x.to(device), test_x.to(device), dev_y, test_y, dev_mask.to(device), test_mask.to(device), dev_attention.to(device), test_attention.to(device), item_num, criterion)

    # グラフ描画用のdf
    if args.map_embedding_per_epoch:
        df = pd.DataFrame(
            columns=['sentence', 'score', 'marker_size', 'epoch', "X", "Y", "Z"])
        from sas.input_data import get_answers
        answers = get_answers(info)

    # ノルム保存用のインスタンス
    norm_dict = Norm()

    for ii in range(args.epochs):
        batch_scoring_loss = 0
        batch_attention_loss = 0
        t0 = time()
        model.train()
        i = item_num

        for batch_x, batch_y, batch_m, batch_a, batch_f in dataloader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            if args.reg:
                batch_y_ps = batch_y[i +
                                     1].type(torch.FloatTensor).abs().to(device)
            else:
                batch_y_ps = batch_y[i +
                                     1].type(torch.LongTensor).abs().to(device)
            batch_f = batch_f[i+1].to(device)
            outputs = model(batch_x, token_type_ids=None,
                            attention_mask=batch_m, labels=batch_y_ps)

            output = outputs[0]
            attn_output = outputs[1]

            loss = 0
            loss_satt = 0

            if args.reg:
                output = torch.squeeze(output)

            if i < len(info.main_factors) and info.satt:
                batch_a_ps = batch_a[i].to(device)
                p_len = attn_output.shape[1]

                loss_satt += attn_loss(attn_output[batch_f],
                                       batch_a_ps[:, :p_len][batch_f])
            if args.plt:
                loss += criterion(output, batch_y_ps, device)
            else:
                loss += criterion(output, batch_y_ps)
            batch_scoring_loss += loss.item() * batch_x.shape[0]

            if info.satt:
                loss += loss_satt * args.lamda
                batch_attention_loss += loss_satt * batch_x.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tr_time = time() - t0
        total_train_time += tr_time

        logger.info(f"Epoch:{ii} Train time :{tr_time:.2f} (s)")

        logger.info(f"Scoring loss: {batch_scoring_loss/len(dataloader):.3f}")
        if info.satt:
            logger.info(
                f"Attention loss: {batch_attention_loss/len(dataloader):.3f}")

        model.eval()
        with torch.no_grad():
            (dev_qwks, test_qwks, dev_mses, test_mses), train_hidden_states, train_golds, (dev_attentions, dev_hidden_states), (test_attentions,
                                                                                                                                test_hidden_states), d_pred, t_pred, dev_target, test_target, (dev_outputs, test_outputs), (_) = eval.evaluate(info, model, eval_dataloader, accuracy_mode=args.accuracy_mode)
            # logger_eval.info(f"[Dev]: {dev_qwk:.3f} [Test]: {test_qwk:.3f}")
            if best <= dev_qwks:
                best = dev_qwks
                test_best = test_qwks
                best_mse = dev_mses
                test_best_mses = test_mses

                b_epoch = ii
                best_test_attentions = test_attentions
                best_test_hidden_states = test_hidden_states
                best_dev_attentions = dev_attentions
                best_dev_hidden_states = dev_hidden_states
                best_dev_pred = d_pred
                best_test_pred = t_pred
                best_dev_outputs = dev_outputs
                best_test_outputs = test_outputs

                # best_dev_norms = torch.norm(best_dev_outputs[2][2], dim=-1)
                # best_test_norms = torch.norm(best_test_outputs[2][2], dim=-1)
                best_dev_norms = None
                best_test_norms = None

                # best_dev_grads_norm = torch.norm(dev_grads, dim=-1)
                # best_test_grads_norm = torch.norm(test_grads, dim=-1)
                best_dev_grads_norm = None
                best_test_grads_norm = None

                if not args.reg:
                    handling_data.save_hidden_states_item(
                        info, train_hidden_states, target=train_golds, prefix="train")
                    handling_data.save_hidden_states_item(
                        info, dev_hidden_states[0], pred=d_pred, attn=best_dev_attentions, norm=best_dev_norms, grad=best_dev_grads_norm, target=dev_target[item_num+1], prefix="dev")
                    handling_data.save_hidden_states_item(
                        info, test_hidden_states[0], pred=t_pred, attn=best_test_attentions, norm=best_test_norms, grad=best_test_grads_norm, target=test_target[item_num + 1], prefix="test")
                if not args.not_save_weights:
                    torch.save({
                        'epoch': ii,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, info.out_dir+"_best_checkpoint.pt")
                    # torch.save(best_model_weight, info.out_dir +
                    #            "_best_model_weights.pt")

                # 答案と単語のベクトルを2次元に圧縮して可視化したグラフを作成
                # to do
            if args.npt is not None:
                log_with_neptune(neptune=neptune, epoch=ii, dev_qwk=dev_qwks, scoring_loss=batch_scoring_loss/len(
                    dataloader), attention_loss=batch_attention_loss/len(dataloader))

            # 描画用のデータをappend
            # if args.map_embedding_per_epoch is True:
            #     assert args.outdim == 2
            #     # train dataを追加
            #     df_train = pd.DataFrame(data={"id": [f"train_{i}" for i in range(train_golds.size(0))], "sentence": answers['train'], "score": list(map(lambda x: f"gold_{x}", train_golds.cpu().detach().numpy().tolist())), "epoch": [ii for _ in range(train_golds.size(0))], "X": train_hidden_states[:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": train_hidden_states[:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     # dev data 追加
            #     df_dev = pd.DataFrame(data={"id": [f"dev_{i}" for i in range(len(d_pred))], "sentence": answers['dev'], "score": list(map(lambda x: f"pred_{x}", dev_target[item_num+1])), "epoch": [ii for _ in range(len(d_pred))], "X": dev_hidden_states[0][:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": dev_hidden_states[0][:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     # test data 追加
            #     df_test = pd.DataFrame(data={"id": [f"test_{i}" for i in range(len(t_pred))], "sentence": answers['test'], "score": list(map(lambda x: f"test_{x}", test_target[item_num+1])), "epoch": [ii for _ in range(len(t_pred))], "X": test_hidden_states[0][:, 0].cpu(
            #     ).detach().numpy().reshape(-1).tolist(), "Y": test_hidden_states[0][:, 1].cpu().detach().numpy().reshape(-1).tolist()})
            #     df = df.append(df_train)
            #     df = df.append(df_dev)
            #     df = df.append(df_test)

            # ノルムをappend
            # norm_dict.append(train_hidden_states.cpu(
            # ).detach().numpy(), 'train')
            # norm_dict.append(
            #     dev_hidden_states[0].cpu().detach().numpy(), 'dev')
            # norm_dict.append(test_hidden_states[0].cpu(
            # ).detach().numpy(), 'test')

    # ノルムを保存
    # norm_dict.save_norm(f"{info.out_dir}_norm.json")

    if args.save_fin:
        handling_data.save_hidden_states_item(
            info, train_hidden_states, target=train_golds, prefix="train_final")
        handling_data.save_hidden_states_item(
            info, dev_hidden_states[0], pred=d_pred, target=dev_target[item_num+1], prefix="dev_final")
        # handling_data.save_hidden_states_metric_item(
        #     info, test_outputs,pred= t_pred,target= test_target[item_num+1], dist=t_dist, ev=t_ev, prefix="test_final")

    # 学習過程のmappingを可視化する
    if args.map_embedding_per_epoch is True:
        from util.mapping.map_embedding_per_epoch import save_map_embedding_per_epoch
        save_map_embedding_per_epoch(
            df, f"{info.out_dir}_embedding_per_epoch_{args.outdim}dim.html")

    logger.info(f"Best epoch {b_epoch}")
    eval.print_info(best, best_mse, item_num, type="Dev", log=logger)
    eval.print_info(test_best, test_mses, item_num, type="Test", log=logger)

    # logger.info(
    #    f"BEST -> [Dev]: {best:.3f} [Test]: {test_best:.3f} ({b_epoch} epoch)")
    if args.npt is not None:
        neptune.log_metric('Test QWK', b_epoch, test_best)
    #
    if args.reg:
        handling_data.save_result_part_scoring_reg(
            info, best_test_pred, test_mask.sum(1), info.test_dir, probs=True, attentions=best_test_attentions, hidden_states=best_test_hidden_states, prefix="test")
        handling_data.save_result_part_scoring_reg(
            info, best_dev_pred, dev_mask.sum(1), info.dev_dir, probs=True, attentions=best_dev_attentions, hidden_states=best_dev_hidden_states, prefix="dev")
    else:
        handling_data.save_result_item_scoring_cls(
            info, best_test_pred, test_mask.sum(1), info.test_dir, item_num, probs=True, attentions=best_test_attentions,  prefix="test", norms=torch.norm(best_test_outputs[2][2], dim=2))
        handling_data.save_result_item_scoring_cls(
            info, best_dev_pred, dev_mask.sum(1), info.dev_dir, item_num, probs=True, attentions=best_dev_attentions,  prefix="dev", norms=torch.norm(best_dev_outputs[2][2], dim=2))

    model.load_state_dict(torch.load(
        info.out_dir + "_best_checkpoint.pt")['model_state_dict'])
    model.eval()
    if not args.reg:
        handling_data.save_train_result_part_scoring_cls_item_scoring(
            info, model, device, eval_dataloader, item_num)

    import cloudpickle
    with open(info.out_dir + '_train_info.pickle', mode='wb') as f:
        cloudpickle.dump(info, f)
    if args.map:
        import util.mapping.map_embedding as map_embedding
        fig = map_embedding.map_embedding(info.out_dir + "_train_result.pickle", info.out_dir +
                                          "_dev_result.pickle", info.train_dir, info.dev_dir, args.item, info.out_dir + "_scatter.html")
        if args.save_fin:
            fig_final = map_embedding.map_embedding(info.out_dir + "_train_final_result.pickle", info.out_dir +
                                                    "_dev_final_result.pickle", info.train_dir, info.dev_dir, args.item, info.out_dir + "_scatter_final.html")
        if args.npt is not None:
            from neptunecontrib.api import log_chart
            log_chart(
                name=f'{info.prompt_name}_{info.item}_e{b_epoch}_scatter', chart=fig)
            if args.save_fin:
                log_chart(
                    name=f'{info.prompt_name}_{info.item}_scatter_final', chart=fig_final)

    if args.output_trustscore:
        from confidence_estimation.trust_score import output_trustscore
        output_trustscore(info.out_dir + "_train_result.pickle", info.out_dir + "_dev_result.pickle",
                          info.dev_dir, info.out_dir + "_confidence_dev.xlsx", info.item_max_score)
        output_trustscore(info.out_dir + "_train_result.pickle", info.out_dir + "_test_result.pickle",
                          info.test_dir, info.out_dir + "_confidence_test.xlsx", info.item_max_score)

    handling_data.save_res(info, b_epoch, best, best_mses,
                           test_best, test_best_mses)

    if args.make_feature_map:
        # best modelをロードする
        from analysis.util import load_model
        model = load_model(info)
        # yだけ全体点の情報も入ってるので修正
        train_y = train_y[item_num + 1]
        dev_y = dev_y[item_num + 1]
        test_y = test_y[item_num + 1]

        # 特徴マップを作成
        from make_feature_map import make_explanation_file
        from interpretability.explanation import Explanation
        explanation = Explanation(model, info)
        dev_explanation_df = make_explanation_file(
            info, explanation, dev_x, dev_y, dev_mask, "dev")
        test_explanation_df = make_explanation_file(
            info, explanation, test_x, test_y, test_mask, "test")

        if args.measure_justification_identification:
            # justification_identificationをする
            from measure_justification_identification import make_justification_identification_file, make_justification_identification_file_correct_miss_separate
            from analysis.util import get_sas_list
            # 訓練データと予測データのjsonファイルを取得
            _, dev_gold_justification_list, dev_gold_score_list, dev_pred_score_list, _ = get_sas_list(
                info, "dev")
            _, test_gold_justification_list, test_gold_score_list, test_pred_score_list, _ = get_sas_list(
                info, "test")
            # 0点を含めるか否かはdmyトークンを使用しているかで決定
            if args.correct_miss_separate:
                make_justification_identification_file_correct_miss_separate(
                    info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, args.dmy, dev_pred_score_list, test_pred_score_list)
            make_justification_identification_file(
                info, dev_gold_justification_list, dev_gold_score_list, dev_explanation_df, test_gold_justification_list, test_gold_score_list, test_explanation_df, args.dmy)

            if args.measure_faithfulness_eraser:
                # eraserでfaithfulnessを計算
                from measure_faithfulness_eraser import make_faithfulness_eraser_file, make_faithfulness_eraser_file_correct_miss_separate
                if args.correct_miss_separate:
                    make_faithfulness_eraser_file_correct_miss_separate(
                        info,  test_pred_score_list, test_explanation_df, test_x, test_y, test_mask, test_attention, model, flip_mode=args.flip_mode)
                make_faithfulness_eraser_file(info,  test_pred_score_list, test_explanation_df,
                                              test_x, test_y, test_mask, test_attention, model, flip_mode=args.flip_mode)

    # reporter.report()


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
