import argparse
import json
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path","-p",dest="path",nargs="*")
    parser.add_argument("--name","-n",dest="name",nargs="*")
    parser.add_argument("--save","-s",dest="s")



    args = parser.parse_args()

    return args


def dipict_graph(jsn_path,save_path,names):
    import plotly.graph_objects as go
    import plotly.io as pio
    fig = go.Figure()
    for p,name in zip(jsn_path,names):
        res = json.load(open(p,"r"))
        qwk_trus_score = res["trustscore"]
        qwk_posterior = res.get("posterior",None)
        qwks = []
        props = []
        for pro,qwk in qwk_trus_score:
            props.append(pro)
            qwks.append(qwk)
        fig.add_trace(go.Scatter(x=props, y=qwks, name=f"{name} TrustScore"))
        qwks = []
        props = []
        if qwk_posterior is not None:
            for pro,qwk in qwk_posterior:
                props.append(pro)
                qwks.append(qwk)
            fig.add_trace(go.Scatter(x=props, y=qwks, name=f"{name} Posterior"))
            # fig.show()
        fig.update_layout(xaxis_title='Proporion of data[%]',
                  yaxis_title='Root Mean Squared Error (RMSE)')
        fig.write_html(save_path + ".html")
        pio.write_image(fig,save_path + ".png")

def main():
    args = parse_args()
    dipict_graph(args.path,args.s,["CrossEntropy","Triplet","Ranked Triplet"])


if __name__ == '__main__':
    main()
