import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd import IGD
from pymoo.termination.ftol import calc_delta_norm
from pymoo.util.normalization import normalize
from pymoo.util.sliding_window import SlidingWindow
from pymoo.visualization.video.callback_video import AnimationCallback
from copy import deepcopy

class RunningMetric(Callback):

    def __init__(self,
                 period=None,
                 indicator="igd") -> None:

        super().__init__()
        self.indicator = indicator

        self.delta_ideal = None
        self.delta_nadir = None
        self.delta_f = None

        self.history = SlidingWindow(period)

    def update(self, algorithm):
        history = self.history

        F = algorithm.opt.get("F")
        c_F, c_ideal, c_nadir = F, F.min(axis=0), F.max(axis=0)

        # find the normalization constant to divide by
        norm = c_nadir - c_ideal

        # make sure all dimensions in ideal are strictly lower than in nadir
        c_nadir[c_ideal == c_nadir] += 1e-16
        norm[c_ideal == c_nadir] = 1.0

        # append the current optimum to the history
        history.append(dict(F=F, ideal=c_ideal, nadir=c_nadir))

        # normalize the current objective space values (use the utopian if equal)
        c_N = normalize(c_F, c_ideal, c_nadir)

        # normalize all previous generations with respect to current ideal and nadir
        N = [normalize(e["F"], c_ideal, c_nadir) for e in history]

        # calculate the delta difference for each previous ideal and nadir point to current
        delta_ideal = [calc_delta_norm(history[k]["ideal"], history[k - 1]["ideal"], norm) for k in
                       range(1, len(history))] + [0.0]
        delta_nadir = [calc_delta_norm(history[k]["nadir"], history[k - 1]["nadir"], norm) for k in
                       range(1, len(history))] + [0.0]

        # now calculate the indicator from each previous one to the current
        if self.indicator == "igd":
            delta_f = [IGD(c_N).do(N[k]) for k in range(len(N))]
        elif self.indicator == "hv":
            hv = Hypervolume(ref_point=np.ones(c_F.shape[1]))
            delta_f = [hv.do(N[k]) for k in range(len(N))]
        else:
            raise Exception("Unknown indicator.")

        self.delta_ideal, self.delta_nadir, self.delta_f = delta_ideal, delta_nadir, delta_f


class RunningMetricAnimation(AnimationCallback):

    def __init__(self,
                 delta_gen,
                 n_plots=4,
                 key_press=True,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.running = RunningMetric()
        self.delta_gen = delta_gen
        self.key_press = key_press
        self.data = SlidingWindow(n_plots)
        self.save_count = 0
        self.fig = None
        self.axes = None   
        self.metric_history = []

    def draw(self, data, ax):

        for tau, x, f, v in data[:-1]:
            ax.plot(x, f, label="t=%s" % tau, alpha=0.6, linewidth=3)

        tau, x, f, v = data[-1]
        ax.plot(x, f, label="t=%s (*)" % tau, alpha=0.9, linewidth=3)

        for k in range(len(v)):
            if v[k]:
                ax.plot([k + 1, k + 1], [0, f[k]], color="black", linewidth=0.5, alpha=0.5)
                ax.plot([k + 1], [f[k]], "o", color="black", alpha=0.5, markersize=2)

        ax.set_yscale("symlog")
        ax.legend()

        ax.set_xlabel("Generation")
        ax.set_ylabel("$\Delta \, f$", rotation=0)

    def update_running_grid(self, output_folder):
        n = self.save_count
        if n == 0:
            return
    
        cols = int(n ** 0.5) + 1
        rows = (n + cols - 1) // cols
        
        matplotlib.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
        })    
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes, dtype=object).reshape(-1)
        
    
        for i in range(n):
            ax = axes[i]
            ax.clear()
            self.draw(self.metric_history[i], ax)  # ← use draw(), not ax.plot()
            ax.set_title(f"({chr(97 + i)})")  # ← (a), (b), (c)...
    
        for j in range(n, len(axes)):
            axes[j].set_visible(False)
    
        plt.tight_layout()
        output_path = os.path.join(output_folder, "runningmetric_final.tiff")
        fig.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
    
    def do(self, _, algorithm, force_plot=False, **kwargs):
        
        running = self.running

        # update the running metric to have the most recent information
        running.update(algorithm)

        tau = algorithm.n_gen

        if (tau > 0 and tau % self.delta_gen == 0) or force_plot:
            self.save_count += 1
            

            f = running.delta_f
            x = np.arange(len(f)) + 1
            v = [max(ideal, nadir) > 0.005 for ideal, nadir in zip(running.delta_ideal, running.delta_nadir)]
            self.data.append((tau, x, f, v))

            self.metric_history.append(deepcopy(list(self.data)))

            
            # Enforce Times New Roman, size 12, black font globally
            matplotlib.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'font.size': 12,
                'text.color': 'black',
                'axes.labelcolor': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
            })
            
            fig, ax = plt.subplots()
            self.draw(self.data, ax)
            output_folder = r"G:\My Drive\CINDY\PHD\PHD 2025\paper 3\Results and diagrams\Images charts" # for saving charts
            output_path = os.path.join(output_folder, f"runningmetric_{self.save_count}.tiff")
            # fig.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
            self.update_running_grid(output_folder)
            
            plt.draw()
            
            if self.key_press:
                def press(event):
                    if event.key == 'q':
                        algorithm.termination.force_termination = True

                fig.canvas.mpl_connect('key_press_event', press)
                
                plt.draw()
                plt.waitforbuttonpress()
                #fig.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.3)
            plt.close('all')
