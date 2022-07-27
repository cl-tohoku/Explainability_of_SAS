import plotly.express as px
import pandas as pd
import plotly.express as px
from collections import defaultdict
from copy import copy, deepcopy

def plot(df, x_column, y_column, animation_frame, animation_group, removed_sentences, save_path=None):

    fig = px.bar(df, x=x_column, y=y_column, animation_frame=animation_frame, animation_group=animation_group,
            color=x_column,  range_y=[0,1.2]).update_layout(title_font={"size":10})

    # for button in fig.layout.updatemenus[0].buttons:
    #     button['args'][1]['frame']['redraw'] = True

    # for k in range(len(fig.frames)):
    #     fig.frames[k]['layout'].update(title_text=f"{df[title_column].iloc[k]}")

    for i, frame in enumerate(fig.frames):
        frame.layout.title = removed_sentences[i]
        
    for step in fig.layout.sliders[0].steps:
        step["args"][1]["frame"]["redraw"] = True

        if save_path is not None:
            fig.write_html(save_path)
        else:
            fig.show()


def make_df_for_plotly(args, flip_probs, sentence, mask = "■"):
    new_dict = defaultdict(list)
    args = copy(args)
    flip_probs = copy(flip_probs)
    sentence = copy(sentence)
    removed_sentences = []

    for  remove_cnt,  (sentence_index , probs) in enumerate(zip(args, flip_probs), start=1):
        if sentence_index is not None:
            sentence[sentence_index] = mask
        removed_sentences.append(' '.join(sentence))
        for label,prob in enumerate(probs):
            new_dict["prob"].append(prob)
            new_dict["label"].append(str(label))
            new_dict["remove_cnt"].append(remove_cnt)
            new_dict["remove_index"].append(sentence_index)

    new_df = pd.DataFrame(new_dict), removed_sentences
    return new_df

def parse_args():
    import argparse
    from os import path
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_info_path", "-info")
    parser.add_argument("--sanity_df", type=path.abspath)
    parser.add_argument("--print-debug-info", "-debug",
                        dest="debug", default=False, action='store_true', help="")
    parser.add_argument('-m', "--method_name",type=str)
    parser.add_argument("--index",type=int, default=0)
    parser.add_argument("-o","--output", type=path.abspath, default=None, help="save path")
    args = parser.parse_args()
    return args

def main():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import pickle
    import pandas as pd
    from analysis.util import load_data
    

    args = parse_args()
    info = pickle.load(open(args.train_info_path, "rb"))
    id2vocab = dict((v,k) for k,v in info.vocab.items())
    df = pickle.load(open(args.sanity_df,'rb'))

    _, _, (test_x, *_) = load_data(info, eval_size=args.index+10)
    sentences = test_x.numpy().tolist()
    sentence = [id2vocab[id]  for id in sentences[args.index] if id not in [info.vocab["<pad>"]]]
    for i in range(20,len(sentence),20):
        sentence[i] = sentence[i] + " <br> "

    query = (df["name"]==args.method_name) & (df["idx"] == args.index)

    removed_args = deepcopy(df[query]["args_most"].iloc[0])
    flip_probs = deepcopy(df[query]["flip_most_probs"].iloc[0])

    new_df, removed_sentences = make_df_for_plotly(removed_args, flip_probs, sentence)

    save_path = f"{info.out_dir}_{args.method_name}_{args.index}_sanitycheck.html" if args.output is None else args.output
    plot(df=new_df, x_column="label", y_column="prob", animation_frame="remove_cnt", animation_group="label", removed_sentences=removed_sentences, save_path=save_path)



if __name__ == "__main__":
    main()