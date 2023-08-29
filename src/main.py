import argparse
import itertools
import os
import sys
import time
import pickle

import dgl
import numpy as np
import torch
from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.egs import EGS
from src.hyperparameter_range import *
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
import scipy.sparse as sp
from collections import OrderedDict


def test(model, history_list, test_list, num_rels, num_nodes, global_graph, use_cuda, all_ans_list, all_ans_r_list, model_name, mode):
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []
    

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        # mp = checkpoint['state_dict']
        # new_mp = OrderedDict()
        # for k, v in mp:
        #     if k not in ['evolve_model.dynamic_emb', 'evolve_model.emb_rel']:
        #         new_mp.update({k : v})
        # model.load_state_dict(new_mp)

        model.load_state_dict(checkpoint['state_dict'], strict=False)


    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    # if args.multi_step:
    #     all_tail_seq = sp.load_npz(
    #         '../data/{}/history/tail_history_{}.npz'.format(args.dataset, history_time_nogt))
    #     # rel
    #     all_rel_seq = sp.load_npz(
    #         '../data/{}/history/rel_history_{}.npz'.format(args.dataset, history_time_nogt))

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, global_graph, test_triples_input)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score, all_ans_r_list[time_idx], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:    
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1
    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, history_len=None, n_layers=None, dropout=None, n_bases=None, angle=None, history_rate=None):
    # load configuration for grid search the best configuration
    if history_len:
        args.train_history_len = history_len
        args.test_history_len = history_len
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases
    if angle:
        args.angle = angle
    if history_rate:
        args.history_rate = history_rate
    mrr_raw = None
    mrr_filter = None
    mrr_raw_r = None
    mrr_filter_r = None
    hit_result_raw = None
    hit_result_filter = None
    hit_result_raw_r = None
    hit_result_filter_r = None

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)   # data class, RGCNLinkDataset
    train_list, train_times = utils.split_by_time(data.train)   # split into time-specific data
    valid_list, valid_times = utils.split_by_time(data.valid)
    test_list, test_times = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels
    if args.dataset == "ICEWS14s":
        num_times = len(train_list) + len(valid_list) + len(test_list) + 1
    else:
        num_times = len(train_list) + len(valid_list) + len(test_list)
    time_interval = train_times[1]-train_times[0]
    print("num_times", num_times, "--------------", time_interval)
    history_val_time_nogt = valid_times[0]
    history_test_time_nogt = test_times[0]
    if args.multi_step:
        print("val only use global history before:", history_val_time_nogt)
        print("test only use global history before:", history_test_time_nogt)

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    local_time = time.localtime()
    model_name = "{}_{}_{}_{}.mdl".format(args.dataset, args.global_gnn, args.encoder, args.decoder)
    model_state_file = "../models/" + model_name
    print("Sanity Check: Cuda: {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_graph = build_sub_graph(num_nodes, num_rels, data.total, use_cuda, args.gpu).to(device)
    model = EGS(global_graph, args.global_gnn, args.global_layers, args.global_heads, num_nodes, num_rels, args.n_hidden, args.task_weight, args.entity_prediction, 
                args.relation_prediction, args.fuse, args.r_fuse, args.n_bases, args.n_basis, args.n_layers, args.dropout, 
                args.self_loop, args.skip_connect, args.encoder, args.decoder, args.opn, args.layer_norm, use_cuda, args.gpu, args.run_analysis).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list, 
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes,
                                                            global_graph, 
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_con = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num+1]
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                        train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]
                # model.global_forward(global_graph)
                _, _, loss, loss_e, loss_r, loss_con = model(history_glist, global_graph, output[0])

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_con.append(loss_con.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                  .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_con), best_mrr, model_name))

            # validation
            if epoch and epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    global_graph, 
                                                                    use_cuda, 
                                                                    all_ans_list_valid, 
                                                                    all_ans_list_r_valid, 
                                                                    model_state_file,  
                                                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_filter
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_filter_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_filter_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            global_graph,
                                                            use_cuda, 
                                                            all_ans_list_test, 
                                                            all_ans_list_r_test, 
                                                            model_state_file, 
                                                            mode="test")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='TKGE')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=50,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph",  action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="history_len,n_layers,dropout,n_bases,angle,history_rate",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    # configuration for global history
    parser.add_argument("--history-rate", type=float, default=0.3,
                        help="history rate")
    parser.add_argument("--global-gnn", type=str, default='rgat', help='type of gnn in global graph')
    parser.add_argument("--global-heads", type=int, default=4, help='heads of attention during RGAT')
    parser.add_argument("--global-layers", type=int, default=2, help='numbers of propagation')
    parser.add_argument("--save", type=str, default="one",
                        help="number of save")

    # configuration for fusion operation
    parser.add_argument("--fuse", type=str, default='gate', help="fusion of global embedding and evolving embedding")
    parser.add_argument("--r-fuse", type=str, default='gate', help="fusion of relation embedding")



    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '../results/{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder+"-"+args.save)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        if args.dataset == "ICEWS14s":
            hp_range_ = hp_range
        if args.dataset == "WIKI":
            hp_range_ = hp_range_WIKI
        if args.dataset == "YAGO":
            hp_range_ = hp_range_YAGO
        if args.dataset == "ICEWS18":
            hp_range_ = hp_range_ICEWS18
        if args.dataset == "ICEWS05-15":
            hp_range_ = hp_range_ICEWS05_15
        if args.dataset == "GDELT":
            hp_range_ = hp_range_GDELT
        grid = hp_range_[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range_[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('\n\n* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            args.test = False
            args.multi_step = False
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            hits = [1, 3, 10]
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))
            # no ground truth
            args.test = True
            args.topk = 0
            args.multi_step = True
            mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r, hit_result_raw, hit_result_filter, hit_result_raw_r, hit_result_filter_r = run_experiment(
                args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3], grid_entry[4], grid_entry[5])
            o_f.write("No ground truth result:\n")
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_raw_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_raw_r[hit_i].item()))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr_filter_r))
            for hit_i, hit in enumerate(hits):
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, hit_result_filter_r[hit_i].item()))

    # single run
    else:
        run_experiment(args)
    sys.exit()



