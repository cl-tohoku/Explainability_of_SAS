import pandas as pd
import plotly.express as px


def save_map_embedding_per_epoch(df, save_path):
    fig = px.scatter(df, x="X", y="Y", animation_frame="epoch", animation_group="id",
                     color="score", hover_name="sentence",
                     size_max=55, range_x=[-10, 10], range_y=[-10, 10])
    # elif args.outdim == 3:
    #     fig = px.scatter_3d(df, x="X", y="Y", z='Z', animation_frame="epoch", animation_group="sentence",
    #                         color="score", hover_name="sentence",
    #                         size="marker_size", range_x=[-5, 5], range_y=[-5, 5], range_z=[-5, 5])
    fig.write_html(save_path)
