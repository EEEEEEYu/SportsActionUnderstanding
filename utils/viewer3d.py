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
seq_path = '/home/matt/projects/vme_research/calibrate/calibrate_data_2025_05_01_data/FAN/sequence_000004'


# Define FLIR camera directory names
FLIR_DIR = 'flir_23604511'
FLIR_FRAME_DIR = 'frame'

# Load flir timestamps
flir_t_dir = os.path.join(seq_path, FLIR_DIR)
flir_t = np.load(os.path.join(flir_t_dir, 't.npy'))
flir_tr = np.load(os.path.join(flir_t_dir, 't_received.npy'))

# Load FLIR frames
flir_frame_dir = os.path.join(seq_path, FLIR_DIR, FLIR_FRAME_DIR)
print(f"Loading FLIR frames from {flir_frame_dir}")

# Get list of npy files in the directory
flir_files = sorted([f for f in os.listdir(flir_frame_dir) if f.endswith('.npy')])
print(f"Found {len(flir_files)} FLIR frame files")

# Load FLIR frames and their timestamps
flir_frames = []

for file in flir_files:
    # Load frame data
    frame_data = np.load(os.path.join(flir_frame_dir, file))
    flir_frames.append(frame_data)

# Convert to numpy arrays for easier manipulation
flir_frames = np.array(flir_frames)

print(f"Loaded {len(flir_frames)} FLIR frames with shape {flir_frames[0].shape}")



# Default directory names
PROPH_DIR = 'proph_00051462'
PROPH_EXP_DIR = 'proph_00051462_exported'
RES_PROPH = (1280, 720)  # Resolution of the prophesee camera

"""
Processes a sequence, extracting events between triggers and
creating event count images
"""

# Load event data
seq_dir = os.path.join(seq_path, PROPH_EXP_DIR)
loader_event = LoadEventStream(seq_dir)

# Load trigger data
trigger_dir = os.path.join(seq_path, PROPH_DIR, 'triggers')

if not os.path.exists(trigger_dir):
    print(f"Trigger directory {trigger_dir} does not exist. Skipping loading triggers.")
    loader_trigger = None
else:
    loader_trigger = Load(trigger_dir)

# Load the event data
print("Loading event data...")
events, event_t = lcd.load_event_data(loader_event)

# Get all triggers
if loader_trigger is not None:
    print("Loading triggers...")
    triggers = np.array(loader_trigger.get_all()['triggers'])
    print(triggers)
    triggers_t = np.array(loader_trigger.get_all()['t']) * 1e6  # Convert to microseconds
    triggers_tr = np.array(loader_trigger.get_all()['t_received'])

    # Ensure we have at least 2 triggers
    if len(triggers_t) < 2:
        print("Not enough triggers found (need at least 2).")

    # Process events between each pair of triggers
    print(f"Found {len(triggers_t)} triggers. Processing events between them...")

    proph0_tr = [tr for tr, t in zip(triggers_tr, triggers) if t == 0]
    proph1_tr = [tr for tr, t in zip(triggers_tr, triggers) if t == 1]

    print(f"Found {len(proph0_tr)} prophesee 0 triggers")
    print(f"Found {len(proph1_tr)} prophesee 1 triggers")


    trigger_events = []
    for i in tqdm.tqdm(range(1, len(triggers))):
        if triggers[i-1] == 1 and triggers[i] == 0:
            # Get events between triggers
            idx = (event_t >= triggers_t[i-1]) & (event_t < triggers_t[i])
            x = events[idx, 0]
            y = events[idx, 1]
            t = event_t[idx]
            p = events[idx, 3]

            trigger_events.append((x, y, t, p))

    print(f"Found {len(trigger_events)} trigger events")


    # Create timestamp arrays with indices
    x_proph0 = range(len(proph0_tr))
    x_proph1 = range(len(proph1_tr))
    x_flir = range(len(flir_tr))

    matched_triggers = []

    # Connect FLIR timestamps to Proph1 timestamps where relevant
    # Find the first proph0 trigger that is closest to the first FLIR timestamp
    flir_idx = 0
    proph_idx = 0
    # Find the proph0 timestamp closest to the first FLIR timestamp
    closest_diff = float('inf')
    for i, t in enumerate(flir_tr):
        diff = abs(t - proph0_tr[0])
        if diff < closest_diff:
            closest_diff = diff
            flir_idx = i

    matched_triggers.append((flir_idx, proph_idx))

    print(f"Closest Proph0 trigger to first FLIR: {proph_idx} with time difference of {closest_diff:.6f}s")

    # Connect the rest of the timestamps
    while flir_idx < len(flir_tr) and proph_idx < len(proph0_tr):
        proph_idx += 1
        flir_idx += 1
        matched_triggers.append((flir_idx, proph_idx))

# TODO: now the FLIR timestamps line up perfectly with the Proph1 timestamps
#  we can use this to synchronize the two cameras


delta_t = 1.0
res_x = 1280
res_y = 720



# Initialize empty lists to store concatenated data
all_events_xy = []
all_events_t = []
all_events_p = []

if loader_trigger is None:
    # If no triggers are found, use all events
    events_xy = np.stack((events[:, 0], events[:, 1]), axis=1)
    events_t = event_t / 1000000  # Convert to seconds
    events_p = events[:, 3]

    all_events_xy.append(events_xy)
    all_events_t.append(events_t)
    all_events_p.append(events_p)
else:
    # Iterate over all triggers and concatenate the arrays
    for trigger_event in trigger_events:
        events_xy = np.stack((trigger_event[0], trigger_event[1]), axis=1)
        events_t = trigger_event[2] / 1000000  # Convert to seconds
        events_p = trigger_event[3]
        
        all_events_xy.append(events_xy)
        all_events_t.append(events_t)
        all_events_p.append(events_p)

# TODO: separate the events into positive and negative

# Concatenate the lists into single arrays
all_events_xy = np.concatenate(all_events_xy, axis=0)
all_events_t = np.concatenate(all_events_t, axis=0)
all_events_p = np.concatenate(all_events_p, axis=0)

print(all_events_p)

all_pos_events_xy = all_events_xy[all_events_p == 1]
all_neg_events_xy = all_events_xy[all_events_p == -1]
all_pos_events_t = all_events_t[all_events_p == 1]
all_neg_events_t = all_events_t[all_events_p == -1]
print('Event Count')
print(all_events_xy.shape[0])
print('Event Count Positive')
print(all_pos_events_xy.shape[0])
print('Event Count Negative')
print(all_neg_events_xy.shape[0])
print('Event Times')
print(np.min(events_t), np.max(events_t))


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
