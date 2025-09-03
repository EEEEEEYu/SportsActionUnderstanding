import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore

from vme_research.hardware.record import Load, LoadEventStream

# load some data
seq_path = '/home/matt/projects/vme_research/calibrate/calibrate_data_2025_08_20_data/procs/sequence_000007/proc/events'

"""
Processes a sequence, extracting events between triggers and
creating event count images
"""

print("Loading event data...")
all_events_xy = np.load(os.path.join(seq_path, 'events_xy.npy'))
all_events_t = np.load(os.path.join(seq_path, 'events_t.npy')) / 1000000
all_events_p = np.load(os.path.join(seq_path, 'events_p.npy'))

# skip the first 10% of events
skip_count = int(0.2 * all_events_xy.shape[0])
all_events_xy = all_events_xy[skip_count:]
all_events_t = all_events_t[skip_count:]
all_events_p = all_events_p[skip_count:]

delta_t = 1.0
res_x = 720
res_y = 720

all_pos_events_xy = all_events_xy[all_events_p == 1]
all_neg_events_xy = all_events_xy[all_events_p == 0]
all_pos_events_t = all_events_t[all_events_p == 1]
all_neg_events_t = all_events_t[all_events_p == 0]
print('Event Count')
print(all_events_xy.shape[0])
print('Event Count Positive')
print(all_pos_events_xy.shape[0])
print('Event Count Negative')
print(all_neg_events_xy.shape[0])
print('Event Times')
print(np.min(all_events_t), np.max(all_events_t))


import matplotlib.pyplot as plt
plt.figure()
plt.hist(all_pos_events_t, bins=100)
plt.title('POS Event Times')
plt.figure()
plt.hist(all_neg_events_t, bins=100)
plt.title('NEG Event Times')
plt.show()

events_pos_t_vis =   10*(((all_pos_events_t - np.min(all_pos_events_t)) / delta_t) - 1.0)
events_pos_x_vis =   10*(all_pos_events_xy[:, 0] / res_x) - 5
events_pos_y_vis =  -10*(all_pos_events_xy[:, 1] / res_x) + 10 * res_y / res_x
events_neg_t_vis =   10*(((all_neg_events_t - np.min(all_neg_events_t)) / delta_t) - 1.0)
events_neg_x_vis =   10*(all_neg_events_xy[:, 0] / res_x) - 5
events_neg_y_vis =  -10*(all_neg_events_xy[:, 1] / res_x) + 10 * res_y / res_x

app = pg.mkQApp("GLScatterPlotItem Example")
w = gl.GLViewWidget()
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')
w.setCameraPosition(distance=20)
w.setBackgroundColor('k')

# setup nearly "orthographic" projection
w.opts['distance'] = 2000
w.opts['fov'] = 1

g = gl.GLGridItem()
w.addItem(g)

pos3 = np.zeros((300000,3))
pos_scatter = gl.GLScatterPlotItem(pos=pos3, color=(0,0,1,.3), size=0.03, pxMode=False)
w.addItem(pos_scatter)
max_events = min(pos3.shape[0], events_pos_t_vis.shape[0])
if max_events < events_pos_t_vis.shape[0]:
    event_indices = np.linspace(0, events_pos_t_vis.shape[0]-1, num=max_events, dtype=np.int32)

    pos3[:max_events, 0] = events_pos_t_vis[event_indices]
    pos3[:max_events, 1] = events_pos_x_vis[event_indices]
    pos3[:max_events, 2] = events_pos_y_vis[event_indices]
    pos3[max_events:, ...] = 0
else:
    pos3[:max_events, 0] = events_pos_t_vis[:max_events]
    pos3[:max_events, 1] = events_pos_x_vis[:max_events]
    pos3[:max_events, 2] = events_pos_y_vis[:max_events]
    pos3[max_events:, ...] = 0
pos_scatter.setData(pos=pos3)

neg3 = np.zeros((300000,3))
neg_scatter = gl.GLScatterPlotItem(pos=pos3, color=(1,0,0,.3), size=0.03, pxMode=False)
w.addItem(neg_scatter)
max_events = min(pos3.shape[0], events_neg_t_vis.shape[0])
if max_events < events_neg_t_vis.shape[0]:
    event_indices = np.linspace(0, events_neg_t_vis.shape[0]-1, num=max_events, dtype=np.int32)

    neg3[:max_events, 0] = events_neg_t_vis[event_indices]
    neg3[:max_events, 1] = events_neg_x_vis[event_indices]
    neg3[:max_events, 2] = events_neg_y_vis[event_indices]
    neg3[max_events:, ...] = 0
else:
    neg3[:max_events, 0] = events_neg_t_vis[:max_events]
    neg3[:max_events, 1] = events_neg_x_vis[:max_events]
    neg3[:max_events, 2] = events_neg_y_vis[:max_events]
    neg3[max_events:, ...] = 0
neg_scatter.setData(pos=neg3)

if __name__ == '__main__':
    pg.exec()
