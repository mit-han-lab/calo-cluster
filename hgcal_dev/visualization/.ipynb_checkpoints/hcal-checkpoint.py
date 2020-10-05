import cycler
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm


class Event():

    def __init__(self, df):
        self.df = df

    def plot_event(self):
        hits = self.df
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(26, 12))
        fig.set_facecolor('white')
        e_max = np.max(hits['energy'].values)
        e_normed = 100.*np.tanh(hits['energy'].values/e_max)

        color_cycler = cycler.cycler(color=['b', 'r', 'g', 'y', 'm', 'c'])()

        for i_cat in range(2):
            color = next(color_cycler)['color']
            cat_mask = hits['hit'] == i_cat
            cat_hits = hits[cat_mask]
            ax0.scatter(cat_hits['x'].values, cat_hits['z'].values, s=(
                e_normed[cat_mask]), c=color)
            ax1.scatter(cat_hits['y'].values, cat_hits['z'].values, s=(
                e_normed[cat_mask]), c=color)
            ax2.scatter(cat_hits['x'].values, cat_hits['y'].values, s=(
                e_normed[cat_mask]), c=color)

        ax0.legend(['Noise', 'Truth'], prop={'size': 24})
        fontsize = 24
        ax0.set_xlabel('x', fontsize=fontsize)
        ax0.set_ylabel('z', fontsize=fontsize)
        ax1.set_xlabel('y', fontsize=fontsize)
        ax1.set_ylabel('z', fontsize=fontsize)
        ax2.set_xlabel('x', fontsize=fontsize)
        ax2.set_ylabel('y', fontsize=fontsize)
        ax0.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='both', which='major', labelsize=20)

    def plot_3d(self):
        hits = self.df
        hits['type'] = 'Truth'
        hits['type'][hits['hit'] == 0] = 'Noise'
        fig = px.scatter_3d(hits, x='x', y='y', z='z',
                            color='type', size='energy')
        return fig
