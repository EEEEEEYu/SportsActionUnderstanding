"""
Coordinate Systems:
- FLIR resolution: 1100x1100 (full resolution after remapping)
- Event resolution: Variable based on data, typically around 700x700 or 1280x720
- Display: Full resolution side-by-side view (events on left, FLIR on right)
- User interactions (line position, rectangles) are in the actual image coordinates
- No scaling needed as we're working with full resolution images
"""

# TODO: save the crop info to crops.json (don't add it to data.json anymore)
#       save name as well into this file
# TODO: update convert_raw.py to read in crops.json and use it when saving
# TODO: update viewer.py to take in right arrow key presses as crop start markers,
#       left arrow key presses as crop end markers,
#       and save out a crop.json file for it

import os
import threading
import multiprocessing
import argparse
import cv2
import json
import numpy as np
import shutil
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, Entry, END, Listbox, Scrollbar, messagebox, StringVar, Checkbutton, BooleanVar, ttk, filedialog
from PIL import Image, ImageTk

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


# binoc
PROPH_DIR = 'proph_00051463'
PROPH_EXPORTED_FOLDER = 'proph_00051463_exported'
METADATA_FILE = 'metadata.json'

# For image data
FLIR_DIR = 'flir_23604512'


class DataRenderer:
    """Renderer that loads and displays FLIR frames and events using the new utils approach"""
    def __init__(self, sequence_path, window_size_ms=33):
        self.sequence_path = sequence_path
        self.window_size_ms = window_size_ms
        self.window_size_us = window_size_ms * 1000  # Convert to microseconds
        
        # Load FLIR frames
        self.frames, self.flir_t, self.flir_res, self.flir_K, self.flir_dist = utils.load_frames(sequence_path)
        self.num_frames = len(self.frames)
        print(f"Loaded {self.num_frames} FLIR frames")
        
        # Load events
        self.events, self.events_t, self.events_res, self.events_K, self.events_dist = utils.load_events(sequence_path)
        print(f"Loaded {len(self.events)} events")
        
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range [0, {self.num_frames})")
        
        # Get FLIR frame
        flir_frame = self.frames[idx]
        if len(flir_frame.shape) == 2:  # Grayscale
            flir_frame = cv2.cvtColor(flir_frame, cv2.COLOR_GRAY2BGR)
        
        # Get corresponding time window for events
        current_time = self.flir_t[idx]
        window_start = current_time
        window_end = current_time + self.window_size_us
        
        # Get events in the time window
        events_in_window = utils.get_events_between(self.events, self.events_t, window_start, window_end)
        
        # Render events
        event_frame = utils.render(events_in_window, self.events_res)
        
        # Apply undistortion/remapping to event count image
        event_frame = utils.undistort_event_count_image(event_frame, self.events_K, self.events_dist, self.events_res)
        
        # Resize event frame to match FLIR frame dimensions if needed
        flir_h, flir_w = flir_frame.shape[:2]
        event_h, event_w = event_frame.shape[:2]
        if (event_h, event_w) != (flir_h, flir_w):
            event_frame = cv2.resize(event_frame, (flir_w, flir_h), interpolation=cv2.INTER_LINEAR)
        
        # Create side-by-side frame
        height = flir_h
        width = flir_w * 2
        
        big_frame = np.zeros((height, width, 3), dtype=np.uint8)
        big_frame[:, :flir_w] = event_frame
        big_frame[:, flir_w:] = flir_frame
        
        # Create overlay
        overlay_frame = cv2.addWeighted(flir_frame, 0.6, event_frame, 0.8, 0)
        
        return big_frame, overlay_frame
    
    def get_time_range(self):
        """Get the time range of the data"""
        return self.flir_t[0], self.flir_t[-1]


class Crop:
    """Class to hold crop information with start/end indices and name"""
    def __init__(self, name, start_index, end_index, zero_rectangles=None):
        self.name = name
        self.start_index = start_index
        self.end_index = end_index
        # List of rectangles to zero out, each rect is (x, y, width, height)
        self.zero_rectangles = zero_rectangles if zero_rectangles is not None else []
    
    def to_dict(self):
        return {
            'name': self.name,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'zero_rectangles': self.zero_rectangles
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            data['name'], 
            data['start_index'], 
            data['end_index'],
            data.get('zero_rectangles', [])
        )
    
    def __str__(self):
        return f"{self.name}: [{self.start_index}, {self.end_index}]"


class CombinedAnnotatorApp:
    def __init__(self, root, base_dir):
        self.root = root
        self.root.title("Combined Image & Event Annotator")
        self.root.resizable(True, True)

        # Keep references to directories
        self.base_dir = base_dir
        
        self.selected_sequence = None
        self.loaded_sequence = None

        # List of crops instead of single start/end
        self.crops = []
        self.temp_start_index = None
        self.temp_end_index = None

        # Store references to the current frames
        self.current_image_frame = None
        self.current_event_frame = None
        self.current_overlay_frame = None
        
        # For tracking which pane is showing
        self.show_overlay = True
        
        # Rectangle drawing state
        self.drawing_mode = False
        self.drawing_rect = False
        self.rect_start = None
        self.current_rect = None
        self.temp_rectangles = []  # Rectangles for current crop being edited
        
        # Store image dimensions for coordinate conversion
        self._image_width = 1200  # Default
        self._image_height = 600   # Default

        # Make rows/columns expand
        # Adjust grid to have 6 rows to accommodate overlay
        for r in range(6):
            self.root.grid_rowconfigure(r, weight=1)
        for c in range(5):
            self.root.grid_columnconfigure(c, weight=1)

        self.current_image_index = 0

        # 1) File Browser
        self.file_browser_frame = Frame(root, bg="darkgray")
        self.file_browser_frame.grid(
            row=0, column=0, rowspan=4, columnspan=1, sticky="nsew", padx=5, pady=5
        )
        self.file_listbox = Listbox(self.file_browser_frame)
        self.file_listbox.pack(side="left", fill="both", expand=False)
        self.scrollbar = Scrollbar(self.file_browser_frame, orient="vertical")
        self.scrollbar.config(command=self.file_listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=self.scrollbar.set)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_folder_select)
        self.populate_file_browser(self.base_dir)

        # 2) Main Image Frame (side-by-side view)
        self.image_frame = Frame(root, bg="lightgray")
        self.image_frame.grid(
            row=0, column=1, rowspan=2, columnspan=2, sticky="nsew", padx=5, pady=5
        )
        self.image_frame.bind("<Configure>", self.on_image_frame_resize)

        self.image_label = Label(self.image_frame, bg="lightgray", text="Side-by-side View")
        self.image_label.pack(fill="both", expand=True)
        
        # Bind mouse events for rectangle drawing
        self.image_label.bind("<Button-1>", self.on_mouse_down)
        self.image_label.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_label.bind("<ButtonRelease-1>", self.on_mouse_up)

        # 3) Overlay Frame (aligned overlay view)
        self.overlay_frame = Frame(root, bg="darkgray")
        self.overlay_frame.grid(
            row=2, column=1, rowspan=2, columnspan=2, sticky="nsew", padx=5, pady=5
        )
        self.overlay_frame.bind("<Configure>", self.on_overlay_frame_resize)
        
        self.overlay_label = Label(self.overlay_frame, bg="darkgray", text="Overlay View")
        self.overlay_label.pack(fill="both", expand=True)

        # 4) Right Frame for Controls
        self.right_frame = Frame(root, bg="darkgray")
        self.right_frame.grid(row=0, column=3, rowspan=4, columnspan=2, sticky="nsew", padx=5, pady=5)

        # --- Crop management controls ---
        Label(self.right_frame, text="Crop Management", font=("Arial", 12, "bold")).pack(pady=(10, 5))
        
        # Temporary start/end controls
        temp_frame = Frame(self.right_frame)
        temp_frame.pack(pady=(5, 10))
        
        self.set_start_button = Button(temp_frame, text="Set Start", command=self.set_temp_start)
        self.set_start_button.grid(row=0, column=0, padx=5)
        
        self.temp_start_label = Label(temp_frame, text="Start: None")
        self.temp_start_label.grid(row=0, column=1, padx=5)
        
        self.set_end_button = Button(temp_frame, text="Set End", command=self.set_temp_end)
        self.set_end_button.grid(row=1, column=0, padx=5, pady=(5, 0))
        
        self.temp_end_label = Label(temp_frame, text="End: None")
        self.temp_end_label.grid(row=1, column=1, padx=5, pady=(5, 0))
        
        # Crop name entry
        Label(self.right_frame, text="Crop Name:").pack(pady=(10, 0))
        self.crop_name_entry = Entry(self.right_frame)
        self.crop_name_entry.bind("<Return>", lambda event: self.add_crop())
        self.crop_name_entry.bind("<Escape>", lambda event: self.root.focus_set())
        self.crop_name_entry.pack(pady=(0, 5))
        
        # Add crop button
        self.add_crop_button = Button(self.right_frame, text="Add Crop", command=self.add_crop)
        self.add_crop_button.pack(pady=(5, 10))
        
        # Crops list
        Label(self.right_frame, text="Crops:", font=("Arial", 10, "bold")).pack()
        crops_frame = Frame(self.right_frame)
        crops_frame.pack(fill="both", expand=True, pady=(5, 10))
        
        self.crops_listbox = Listbox(crops_frame, height=6)
        self.crops_listbox.pack(side="left", fill="both", expand=True)
        
        crops_scrollbar = Scrollbar(crops_frame, orient="vertical")
        crops_scrollbar.config(command=self.crops_listbox.yview)
        crops_scrollbar.pack(side="right", fill="y")
        self.crops_listbox.config(yscrollcommand=crops_scrollbar.set)
        
        # Delete crop buttons
        delete_frame = Frame(self.right_frame)
        delete_frame.pack(pady=(0, 20))
        
        self.delete_crop_button = Button(delete_frame, text="Delete Selected", command=self.delete_crop)
        self.delete_crop_button.pack(side="left", padx=5)
        
        self.delete_all_crops_button = Button(delete_frame, text="Delete All", command=self.delete_all_crops)
        self.delete_all_crops_button.pack(side="left", padx=5)
        
        # --- Rectangle drawing controls ---
        Label(self.right_frame, text="Zero Regions", font=("Arial", 10, "bold")).pack()
        
        # Rectangle drawing mode button
        self.rect_mode_button = Button(
            self.right_frame, 
            text="Draw Rectangle Mode", 
            command=self.toggle_drawing_mode,
            bg="lightgray"
        )
        self.rect_mode_button.pack(pady=(5, 5))
        
        # Clear rectangles button
        self.clear_rects_button = Button(
            self.right_frame, 
            text="Clear All Rectangles", 
            command=self.clear_rectangles
        )
        self.clear_rects_button.pack(pady=(0, 10))
        
        # Rectangle list display
        self.rect_listbox = Listbox(self.right_frame, height=4)
        self.rect_listbox.pack(fill="x", padx=5, pady=(0, 20))

        # Fixed window duration for events (hidden from UI)
        self.window_duration_ms = 33  # Default 33ms

        # --- Downsampling control ---
        self.downsample_var = BooleanVar(value=True)  # Default to enabled
        self.downsample_checkbox = Checkbutton(
            self.right_frame, 
            text="Downsample Events (faster)", 
            variable=self.downsample_var,
            command=self.on_downsample_change
        )
        self.downsample_checkbox.pack(pady=(10, 10))

        # Video export button
        video_export_button = Button(
            self.right_frame,
            text="Export Video",
            command=self.export_video,
            bg="lightblue"
        )
        video_export_button.pack(side="bottom", fill="x", padx=5, pady=5)
        
        load_button = Button(
            self.right_frame, 
            text="Load", 
            command=lambda: self.on_load_button(self.selected_sequence)
        )
        load_button.pack(side="bottom", fill="x", padx=5, pady=5)
        
        save_button = Button(
            self.right_frame, 
            text="Save", 
            command=lambda: self.save_sequence(self.loaded_sequence)
        )
        save_button.pack(side="bottom", fill="x", padx=5, pady=5)

        # 5) Slider (along the bottom)
        self.slider = Scale(
            root,
            from_=0,
            to=0,
            orient=HORIZONTAL,
            command=self.update_display,
            length=600
        )
        self.slider.grid(row=5, column=0, columnspan=5, sticky="ew", padx=5, pady=5)

        # Loading label and progress bar
        self.loading_label = Label(self.image_frame, text="Loading...", bg="gray", fg="white")
        self.progress_frame = Frame(self.image_frame, bg="gray")
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_label = Label(self.progress_frame, text="", bg="gray", fg="white")
        self.progress_bar.pack(pady=5)
        self.progress_label.pack()

        # Initialize renderer
        self.data_renderer = None

        # Bind arrow keys to navigate images
        self.root.bind("<Left>", self.prev_image_fine)
        self.root.bind("<Right>", self.next_image_fine)
        self.root.bind("<Down>", self.next_image_coarse)
        self.root.bind("<Up>", self.prev_image_coarse)
        self.root.bind("a", self.set_temp_start)
        self.root.bind("d", self.set_temp_end)
        self.root.bind("w", self.focus_crop_name)

    def populate_file_browser(self, directory):
        """Populate the file browser with folders in the given directory."""
        self.file_listbox.delete(0, END)
        if os.path.isdir(directory):
            for item in sorted(os.listdir(directory)):
                fullpath = os.path.join(directory, item)
                if os.path.isdir(fullpath):
                    self.file_listbox.insert(END, item)
                    self.file_listbox.itemconfig(END, {'fg': 'orange'})

                    # Check if exported folder exists
                    proph_exported = os.path.join(fullpath, PROPH_EXPORTED_FOLDER)
                    if not os.path.exists(proph_exported):
                        self.file_listbox.itemconfig(END, {'fg': 'red'})
                    else:
                        if os.path.exists(os.path.join(proph_exported, METADATA_FILE)):
                            self.file_listbox.itemconfig(END, {'fg': 'black'})

    def on_folder_select(self, event):
        selected_index = self.file_listbox.curselection()
        if selected_index:
            selected_folder = self.file_listbox.get(selected_index)
            new_directory = os.path.join(self.base_dir, selected_folder)
            self.selected_sequence = new_directory

    def save_sequence(self, seq_dir):
        if not self.crops:
            messagebox.showwarning("No Crops", "Please add at least one crop before saving.")
            return
            
        print(f'Saving {len(self.crops)} crop metadata...')
        
        if self.data_renderer is None:
            messagebox.showwarning("No Data", "Please load data before saving.")
            return
        
        # Process each crop and prepare crop metadata
        crops_metadata = []
        for crop in self.crops:
            # Get FLIR timestamps for the crop
            flir_start_time = self.data_renderer.flir_t[crop.start_index]
            flir_end_time = self.data_renderer.flir_t[crop.end_index]
            
            # Get event timestamps for the crop time range
            # Events are already in microseconds
            events_in_range = utils.get_events_between(
                self.data_renderer.events,
                self.data_renderer.events_t,
                flir_start_time,
                flir_end_time
            )
            
            if len(events_in_range) > 0:
                event_start_time = float(events_in_range[0, 2])  # First event timestamp
                event_end_time = float(events_in_range[-1, 2])   # Last event timestamp
            else:
                # No events in this range, use FLIR times
                event_start_time = flir_start_time
                event_end_time = flir_end_time
            
            crop_metadata = {
                'name': crop.name,
                'start_index': crop.start_index,
                'end_index': crop.end_index,
                'flir_start_time': float(flir_start_time),  # Convert to float for JSON
                'flir_end_time': float(flir_end_time),
                'event_start_time': event_start_time,
                'event_end_time': event_end_time,
                'zero_rectangles': crop.zero_rectangles
            }
            crops_metadata.append(crop_metadata)
        
        # Save crop.json in the root folder of the sequence
        crop_json_path = os.path.join(seq_dir, 'crop.json')
        with open(crop_json_path, 'w') as f:
            json.dump({
                'crops': crops_metadata,
                'window_duration_ms': self.window_duration_ms
            }, f, indent=2)
        print(f"Saved crop.json with {len(crops_metadata)} crops to {crop_json_path}")
        
        messagebox.showinfo("Save Complete", f"Saved {len(crops_metadata)} crop(s) to crop.json")
        self.populate_file_browser(self.base_dir)

    def on_load_button(self, seq_dir):
        if self.loaded_sequence:
            confirm = messagebox.askyesno(
                "Confirm Load",
                "A sequence is already loaded. Do you want to load a new sequence?"
            )
            if not confirm:
                return
        self.load_sequence(seq_dir)

    def load_sequence(self, seq_dir):
        self.loading_label.config(text="Loading data...")
        self.loading_label.place(relx=0.5, rely=0.4, anchor="center")
        self.progress_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.image_label.config(image="", text="")
        self.overlay_label.config(image="", text="")

        def _load_data():
            try:
                # Check if processed data exists
                proc_dir = os.path.join(seq_dir, 'proc')
                if not os.path.exists(proc_dir):
                    self.root.after(0, lambda: messagebox.showerror(
                        "Data Not Found", 
                        f"Processed data not found in {seq_dir}\nPlease ensure 'proc' directory exists with FLIR and event data."
                    ))
                    self.root.after(0, lambda: self.loading_label.place_forget())
                    self.root.after(0, lambda: self.progress_frame.place_forget())
                    return
                
                # Load data using the new approach
                self.root.after(0, lambda: self.loading_label.config(text="Loading FLIR and event data..."))
                
                self.data_renderer = DataRenderer(seq_dir, window_size_ms=self.window_duration_ms)
                
                # Check crop.json for settings and crops
                crop_file = os.path.join(seq_dir, 'crop.json')
                if os.path.exists(crop_file):
                    with open(crop_file, 'r') as f:
                        crop_data = json.load(f)
                        print('Loading crop data...', crop_data)
                        
                        # Load settings
                        if 'window_duration_ms' in crop_data:
                            self.window_duration_ms = crop_data['window_duration_ms']
                            self.data_renderer.window_size_ms = self.window_duration_ms
                            self.data_renderer.window_size_us = self.window_duration_ms * 1000
                        
                        # Load crops
                        if 'crops' in crop_data:
                            self.crops = [Crop.from_dict(crop) for crop in crop_data['crops']]
                            self.update_crops_display()
                
                num_frames = len(self.data_renderer)
                print(f"Loaded {num_frames} frames")
                
                self.root.title(f"Combined Annotator - {seq_dir}")
                self.loaded_sequence = seq_dir
                self.root.after(0, lambda: self.finish_loading(num_frames))
                
            except Exception as e:
                print(f"Error loading data: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: messagebox.showerror("Load Error", f"Failed to load data: {str(e)}"))
                self.root.after(0, lambda: self.loading_label.place_forget())
                self.root.after(0, lambda: self.progress_frame.place_forget())

        threading.Thread(target=_load_data, daemon=True).start()

    def finish_loading(self, num_frames):
        self.loading_label.place_forget()
        self.progress_frame.place_forget()
        self.progress_bar.config(value=0)

        if num_frames > 0:
            self.slider.config(to=num_frames - 1)
            self.update_display(0)
        else:
            self.slider.config(to=0)
            self.image_label.config(text="No data loaded")
            self.overlay_label.config(text="No data loaded")
            # Event display removed

    def update_display(self, index):
        """Update both image and event displays"""
        if not self.data_renderer:
            return

        index = int(index)
        max_index = len(self.data_renderer) - 1
        
        if 0 <= index <= max_index:
            self.current_image_index = index
            
            # Get side-by-side frame and overlay
            image_frame, overlay_frame = self.data_renderer[index]
            self.current_image_frame = image_frame
            self.current_overlay_frame = overlay_frame
            
            # Draw both views
            self.draw_image()
            self.draw_overlay()
            
            self.slider.set(index)

    def draw_image(self):
        """Draw the RGB image with rectangles and vertical line"""
        if self.current_image_frame is None:
            return
        
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()

        if frame_width < 1 or frame_height < 1:
            return

        # Create a copy to draw on
        display_frame = self.current_image_frame.copy()
        
        # Get image dimensions
        img_height, img_width = display_frame.shape[:2]
        
        # Store original image dimensions for coordinate conversion
        self._image_width = img_width
        self._image_height = img_height
        
        # Calculate the boundary between event and FLIR images
        # The FullResolutionBSRenderer places them side by side
        # We need to determine where the event image ends and FLIR begins
        # This is approximately half the width, but we should be more precise
        event_width = img_width // 2  # Approximate for now
        
        # Draw all saved rectangles for this frame (only on FLIR side)
        for rect in self.temp_rectangles:
            x, y, w, h = rect
            # Rectangle coordinates are for FLIR image, so add event_width offset
            cv2.rectangle(display_frame, (event_width + x, y), (event_width + x + w, y + h), (0, 0, 255), 2)
        
        # Draw current rectangle being drawn (only on FLIR side)
        if self.current_rect:
            x, y, w, h = self.current_rect
            cv2.rectangle(display_frame, (event_width + x, y), (event_width + x + w, y + h), (255, 0, 0), 2)
        
        # Apply downsampling if enabled
        if self.downsample_var.get():
            # Downsample the image to half resolution first
            img_height, img_width = display_frame.shape[:2]
            downsampled = cv2.resize(
                display_frame,
                (img_width // 2, img_height // 2),
                interpolation=cv2.INTER_AREA
            )
            # Then resize to fit frame
            resized_cv = cv2.resize(
                downsampled,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR  # Faster for already downsampled images
            )
        else:
            # Full resolution - resize directly to frame
            resized_cv = cv2.resize(
                display_frame, 
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA
            )
        
        pil_img = Image.fromarray(cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB))
        self.img_tk = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.img_tk, text="")
        self.image_label.image = self.img_tk



    def on_image_frame_resize(self, event):
        self.draw_image()
    
    def on_overlay_frame_resize(self, event):
        self.draw_overlay()
    
    def draw_overlay(self):
        """Draw the overlay image"""
        if self.current_overlay_frame is None:
            return
        
        frame_width = self.overlay_frame.winfo_width()
        frame_height = self.overlay_frame.winfo_height()
        
        if frame_width < 1 or frame_height < 1:
            return
        
        # Create a copy to draw on
        display_frame = self.current_overlay_frame.copy()
        
        # Get overlay dimensions
        img_height, img_width = display_frame.shape[:2]
        
        # Draw rectangles on overlay
        for rect in self.temp_rectangles:
            x, y, w, h = rect
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Draw current rectangle being drawn
        if self.current_rect:
            x, y, w, h = self.current_rect
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Apply downsampling if enabled
        if self.downsample_var.get():
            # Downsample the image to half resolution first
            downsampled = cv2.resize(
                display_frame,
                (img_width // 2, img_height // 2),
                interpolation=cv2.INTER_AREA
            )
            # Then resize to fit frame
            resized_cv = cv2.resize(
                downsampled,
                (frame_width, frame_height),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            # Full resolution - resize directly to frame
            resized_cv = cv2.resize(
                display_frame, 
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA
            )
        
        pil_img = Image.fromarray(cv2.cvtColor(resized_cv, cv2.COLOR_BGR2RGB))
        self.overlay_img_tk = ImageTk.PhotoImage(pil_img)
        self.overlay_label.config(image=self.overlay_img_tk, text="")
        self.overlay_label.image = self.overlay_img_tk


    def _is_entry_focused(self):
        """Check if any Entry widget has focus"""
        focused_widget = self.root.focus_get()
        return isinstance(focused_widget, Entry)

    def next_image_fine(self, event=None):
        """Move to the next image."""
        if self._is_entry_focused():
            return
        max_index = len(self.data_renderer) - 1 if self.data_renderer else 0
        if self.current_image_index < max_index:
            self.update_display(self.current_image_index + 1)

    def next_image_coarse(self, event=None):
        """Jump forward by 10 frames."""
        if self._is_entry_focused():
            return
        max_index = len(self.data_renderer) - 1 if self.data_renderer else 0
        if self.current_image_index < max_index - 10:
            self.update_display(self.current_image_index + 10)
        else:
            self.update_display(max_index)

    def prev_image_fine(self, event=None):
        """Move to the previous image."""
        if self._is_entry_focused():
            return
        if self.current_image_index > 0:
            self.update_display(self.current_image_index - 1)

    def prev_image_coarse(self, event=None):
        """Jump backward by 10 frames."""
        if self._is_entry_focused():
            return
        if self.current_image_index >= 10:
            self.update_display(self.current_image_index - 10)
        else:
            self.update_display(0)

    def set_temp_start(self, event=None):
        """Set temporary start index for a new crop"""
        if self._is_entry_focused():
            return
        self.temp_start_index = self.current_image_index
        self.temp_start_label.config(text=f"Start: {self.temp_start_index}")
    
    def set_temp_end(self, event=None):
        """Set temporary end index for a new crop"""
        if self._is_entry_focused():
            return
        self.temp_end_index = self.current_image_index
        self.temp_end_label.config(text=f"End: {self.temp_end_index}")
    
    def focus_crop_name(self, event=None):
        """Focus the crop name entry field"""
        if self._is_entry_focused():
            return
        self.crop_name_entry.focus_set()
    
    def add_crop(self):
        """Add a new crop with the current temporary start/end"""
        if self.temp_start_index is None or self.temp_end_index is None:
            messagebox.showwarning("Incomplete Crop", "Please set both start and end indices.")
            return
        
        if self.temp_start_index > self.temp_end_index:
            messagebox.showwarning("Invalid Range", "Start index must be before end index.")
            return
        
        crop_name = self.crop_name_entry.get().strip()
        if not crop_name:
            # Auto-generate name like crop0, crop1, etc.
            crop_name = f"crop{len(self.crops)}"
        
        # Check for duplicate names
        for crop in self.crops:
            if crop.name == crop_name:
                messagebox.showwarning("Duplicate Name", f"A crop named '{crop_name}' already exists.")
                return
        
        # Add the new crop with rectangles
        new_crop = Crop(
            crop_name, 
            self.temp_start_index, 
            self.temp_end_index, 
            self.temp_rectangles.copy()
        )
        self.crops.append(new_crop)
        self.update_crops_display()
        
        # Clear the temp rectangles for next crop
        self.temp_rectangles = []
        self.update_rect_display()
        
        # Set next start to be end+1 of this crop
        next_start = self.temp_end_index + 1
        max_index = len(self.data_renderer) - 1 if self.data_renderer else 0
        if next_start <= max_index:
            self.temp_start_index = next_start
            self.temp_start_label.config(text=f"Start: {self.temp_start_index}")
        else:
            self.temp_start_index = None
            self.temp_start_label.config(text="Start: None")
        
        # Clear end and name
        self.temp_end_index = None
        self.temp_end_label.config(text="End: None")
        self.crop_name_entry.delete(0, END)
        
        # Defocus the entry field
        self.root.focus_set()
    
    def delete_crop(self):
        """Delete the selected crop"""
        selection = self.crops_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a crop to delete.")
            return
        
        index = selection[0]
        crop = self.crops[index]
        
        confirm = messagebox.askyesno("Confirm Delete", f"Delete crop '{crop.name}'?")
        if confirm:
            del self.crops[index]
            self.update_crops_display()
    
    def delete_all_crops(self):
        """Delete all crops"""
        if not self.crops:
            messagebox.showinfo("No Crops", "There are no crops to delete.")
            return
        
        confirm = messagebox.askyesno("Confirm Delete All", 
                                    f"Delete all {len(self.crops)} crops? This cannot be undone.")
        if confirm:
            self.crops = []
            self.update_crops_display()
            # Clear temp rectangles as well
            self.temp_rectangles = []
            self.update_rect_display()
            self.draw_image()
    
    def update_crops_display(self):
        """Update the crops listbox display"""
        self.crops_listbox.delete(0, END)
        for crop in self.crops:
            self.crops_listbox.insert(END, str(crop))

    def on_downsample_change(self):
        """Handle downsample checkbox change"""
        if self.loaded_sequence and self.data_renderer:
            # Downsampling is now handled in the drawing functions
            self.draw_image()
            self.draw_overlay()
    
    def toggle_drawing_mode(self):
        """Toggle rectangle drawing mode"""
        self.drawing_mode = not self.drawing_mode
        if self.drawing_mode:
            self.rect_mode_button.config(bg="lightgreen", text="Exit Drawing Mode")
            self.image_label.config(cursor="crosshair")
        else:
            self.rect_mode_button.config(bg="lightgray", text="Draw Rectangle Mode")
            self.image_label.config(cursor="")
            self.drawing_rect = False
            self.current_rect = None
            self.draw_image()  # Redraw to remove any temporary rectangle
    
    def clear_rectangles(self):
        """Clear all temporary rectangles"""
        self.temp_rectangles = []
        self.update_rect_display()
        self.draw_image()
    
    def on_mouse_down(self, event):
        """Handle mouse down event for rectangle drawing"""
        if not self.drawing_mode or self.current_image_frame is None:
            return
        
        # Get image dimensions and calculate scaling
        img_height, img_width = self.current_image_frame.shape[:2]
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        # Convert click coordinates to image coordinates
        x = int(event.x * img_width / label_width)
        y = int(event.y * img_height / label_height)
        
        # Calculate event width
        event_width = self._image_width // 2
        
        # Only allow rectangle drawing on FLIR side (right half)
        if x < event_width:
            return
        
        # Adjust x coordinate to be relative to FLIR image
        x = x - event_width
        
        self.rect_start = (x, y)
        self.drawing_rect = True
    
    def on_mouse_drag(self, event):
        """Handle mouse drag event for rectangle drawing"""
        if not self.drawing_mode or not self.drawing_rect or self.current_image_frame is None:
            return
        
        # Get image dimensions and calculate scaling
        img_height, img_width = self.current_image_frame.shape[:2]
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        # Convert current mouse position to image coordinates
        x = int(event.x * img_width / label_width)
        y = int(event.y * img_height / label_height)
        
        # Calculate event width
        event_width = self._image_width // 2
        
        # Clamp to FLIR image bounds (right half)
        x = max(event_width, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        
        # Adjust x coordinate to be relative to FLIR image
        x = x - event_width
        
        # Update current rectangle
        x1, y1 = self.rect_start
        x2, y2 = x, y
        
        # Ensure proper ordering
        self.current_rect = (
            min(x1, x2), min(y1, y2),
            abs(x2 - x1), abs(y2 - y1)
        )
        
        self.draw_image()  # Redraw with temporary rectangle
    
    def on_mouse_up(self, event):
        """Handle mouse up event for rectangle drawing"""
        if not self.drawing_mode or not self.drawing_rect or not self.current_rect:
            return
        
        # Add rectangle if it has non-zero area
        x, y, w, h = self.current_rect
        if w > 0 and h > 0:
            self.temp_rectangles.append(self.current_rect)
            self.update_rect_display()
        
        self.drawing_rect = False
        self.current_rect = None
        self.draw_image()
    
    def update_rect_display(self):
        """Update the rectangle listbox display"""
        self.rect_listbox.delete(0, END)
        for i, rect in enumerate(self.temp_rectangles):
            x, y, w, h = rect
            self.rect_listbox.insert(END, f"Rect {i+1}: ({x},{y}) {w}x{h}")
    
    
    
    def export_video(self):
        """Export the current visualization as a video file"""
        if not self.data_renderer:
            messagebox.showwarning("No Data", "Please load a sequence before exporting video.")
            return
        
        # Ask user which view to export
        from tkinter import Toplevel, Radiobutton, IntVar
        
        dialog = Toplevel(self.root)
        dialog.title("Export Video Options")
        dialog.geometry("400x300")
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # View selection
        Label(dialog, text="Select view to export:", font=("Arial", 12)).pack(pady=10)
        
        view_var = IntVar(value=1)
        Radiobutton(dialog, text="Side-by-side View", variable=view_var, value=1).pack(pady=5)
        Radiobutton(dialog, text="Overlay View", variable=view_var, value=2).pack(pady=5)
        
        # Frame range selection
        Label(dialog, text="Frame Range:", font=("Arial", 12)).pack(pady=(20, 5))
        
        range_frame = Frame(dialog)
        range_frame.pack(pady=5)
        
        Label(range_frame, text="Start:").grid(row=0, column=0, padx=5)
        start_entry = Entry(range_frame, width=10)
        start_entry.insert(0, "0")
        start_entry.grid(row=0, column=1, padx=5)
        
        Label(range_frame, text="End:").grid(row=0, column=2, padx=5)
        end_entry = Entry(range_frame, width=10)
        max_frame = len(self.data_renderer) - 1
        end_entry.insert(0, str(max_frame))
        end_entry.grid(row=0, column=3, padx=5)
        
        # FPS selection
        Label(dialog, text="FPS:", font=("Arial", 12)).pack(pady=(20, 5))
        fps_entry = Entry(dialog, width=10)
        fps_entry.insert(0, "30")
        fps_entry.pack()
        
        # Button frame
        button_frame = Frame(dialog)
        button_frame.pack(pady=20)
        
        def on_export():
            try:
                start_frame = int(start_entry.get())
                end_frame = int(end_entry.get())
                fps = int(fps_entry.get())
                view_type = "side_by_side" if view_var.get() == 1 else "overlay"
                
                # Validate inputs
                if start_frame < 0 or end_frame > max_frame or start_frame > end_frame:
                    messagebox.showerror("Invalid Range", f"Please enter a valid frame range (0-{max_frame})")
                    return
                
                if fps <= 0 or fps > 120:
                    messagebox.showerror("Invalid FPS", "Please enter a valid FPS value (1-120)")
                    return
                
                dialog.destroy()
                self._perform_video_export(view_type, start_frame, end_frame, fps)
                
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numbers")
        
        Button(button_frame, text="Export", command=on_export, bg="lightgreen").pack(side="left", padx=10)
        Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=10)
        
        # Focus on the dialog
        dialog.focus_set()
    
    def _perform_video_export(self, view_type, start_frame, end_frame, fps):
        """Perform the actual video export"""
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        # Show progress dialog
        self.loading_label.config(text="Exporting video...")
        self.loading_label.place(relx=0.5, rely=0.4, anchor="center")
        self.progress_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        def _export():
            try:
                # Get frame dimensions based on view type
                if view_type == "side_by_side":
                    sample_frame, _ = self.data_renderer[start_frame]
                else:  # overlay
                    _, sample_frame = self.data_renderer[start_frame]
                
                height, width = sample_frame.shape[:2]
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    raise Exception("Failed to open video writer")
                
                # Export frames
                total_frames = end_frame - start_frame + 1
                for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                    # Update progress
                    progress = int((i / total_frames) * 100)
                    self.root.after(0, lambda p=progress, i=i, t=total_frames: [
                        self.progress_bar.config(value=p),
                        self.progress_label.config(text=f"Writing frame {i+1}/{t}")
                    ])
                    
                    # Get the frame
                    if view_type == "side_by_side":
                        frame, _ = self.data_renderer[frame_idx]
                    else:  # overlay
                        _, frame = self.data_renderer[frame_idx]
                    
                    # Apply rectangles and vertical line if needed
                    display_frame = frame.copy()
                    
                    # Find which crop this frame belongs to (if any)
                    current_crop = None
                    for crop in self.crops:
                        if crop.start_index <= frame_idx <= crop.end_index:
                            current_crop = crop
                            break
                    
                    if current_crop:
                        # Draw rectangles
                        if current_crop.zero_rectangles:
                            for rect in current_crop.zero_rectangles:
                                x, y, w, h = rect
                                if view_type == "side_by_side":
                                    # Draw only on FLIR side
                                    event_width = width // 2
                                    cv2.rectangle(display_frame, (event_width + x, y), 
                                                (event_width + x + w, y + h), (0, 0, 255), 2)
                                else:
                                    # Draw on overlay
                                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    
                    # Write frame
                    out.write(display_frame)
                
                # Release video writer
                out.release()
                
                # Show success message
                self.root.after(0, lambda: [
                    self.loading_label.place_forget(),
                    self.progress_frame.place_forget(),
                    self.progress_bar.config(value=0),
                    messagebox.showinfo("Export Complete", f"Video exported successfully to:\n{file_path}")
                ])
                
            except Exception as e:
                self.root.after(0, lambda: [
                    self.loading_label.place_forget(),
                    self.progress_frame.place_forget(),
                    self.progress_bar.config(value=0),
                    messagebox.showerror("Export Failed", f"Failed to export video:\n{str(e)}")
                ])
        
        # Run export in thread
        threading.Thread(target=_export, daemon=True).start()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/matt/DATA/LCD_SYNC')
    args = parser.parse_args()
    
    root = Tk()
    root.geometry("1800x1400")
    app = CombinedAnnotatorApp(root, args.dataset_dir)
    root.mainloop()