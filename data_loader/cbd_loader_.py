# Imports data from Ptychography experiments at CID, petra 3
# 
# Author: Ahmed H. Mokhtar 
# Email: ahmed.mokhtar@desy.de
# Date : Nov 2025

import numpy as np 
import h5py
import copy
import json
from dataclasses import dataclass, field, fields
import re

import glob
import os 
from .utils import time_it, parse_log, energy2wavelength_a, log_roi_params, inject_attrs, wavelength_a2energy
from . import utils as ut

@dataclass
class Exp:
    year: int
    beamtime_id: str
    # energy: float # keV
    beamline: str = "p11"
    det_distance: float = None   # microns

    def __post_init__(self):
        base = f"/asap3/petra3/gpfs/{self.beamline}/{self.year}/data/{self.beamtime_id}/raw"
        self.scan_frames_dir = os.path.join(base, "scan_frames")
        self.logs_dir = os.path.join(base, "server_log", "Scan_logs")
        
    # ------------ directories -----------
    

@dataclass
class Scan:
    """
    A scan consists of:
      - metadata      (general info from the log header)
      - scan_defs     (per-scan info: points, step counts, start/end)
      - attributes    (per-point arrays: motor positions, exposure time, etc.)
      - file_list     (list of h5/nxs detector files)
    """

    exp: Exp
    scan_num: int
    file_list: list = None
    
    detector: str = field(default = 'Eiger', repr = True)
    sample_name: str = field(default = None, repr = True)
    
    def __post_init__(self):
        
        # ----------------------------
        # Parse Log File
        log_path = self.log_file(self.scan_num) 
        print(log_path)
        with open(log_path, 'r') as log_file:
            log_str = ''
            for line in log_file:
                if line.startswith('# '):
                    log_str += line.strip('# ')
                    
        _, scan_defs, _ = parse_log(log_str)
        
        for scan_name, scan_dict in scan_defs.items():
            inject_attrs(self, scan_dict, scan_name)
            
        # ______________ Data Dir ______________
        self.data_dir = self.scan_dir(self.scan_num)
        
        print(f"Data Directory is : {self.data_dir}")
        
        files_h5  = sorted(glob.glob(os.path.join(self.data_dir, "*.h5")))
        files_nxs = sorted(glob.glob(os.path.join(self.data_dir, "*.nxs")))

        self.file_list = files_h5 if len(files_h5) else files_nxs

        if len(self.file_list) == 0:
            raise FileNotFoundError(f"No .h5 or .nxs files found in {data_dir}")

        if all(hasattr(self, x) for x in ["scan_x_start_point", "scan_x_end_point", "scan_x_steps_count"]):
            self.step_size_x = (self.scan_x_end_point - self.scan_x_start_point) / self.scan_x_steps_count

        if all(hasattr(self, x) for x in ["scan_y_start_point", "scan_y_end_point", "scan_y_steps_count"]):
            self.step_size_y = (self.scan_y_end_point - self.scan_y_start_point) / self.scan_y_steps_count
        
        # _________  Master File ________
        master_file = self.master_file(self.scan_num)

        with h5py.File(master_file, "r") as f:
            self.wavelength = f["/entry/instrument/beam/incident_wavelength"][...] * ut.UNIT_MAP['Angs']
            self.energy = wavelength_a2energy(self.wavelength)

    
    def __repr__(self):
        class_name = type(self).__name__
        sample_name = self.sample_name if self.sample_name is not None else ''
        
        return f"{class_name} | sample: {sample_name} | step_size={self.step_size_x, self.step_size_y}, step_points = {self.scan_x_steps_count, self.scan_y_steps_count}"
        
    def scan_dir(self, scan_num):
        """Returns full directory for a particular scan number."""
        if self.detector == "Eiger":
            return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}")
        
        else:
            return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}_{self.detector}")

    def log_file(self, scan_num):
        return os.path.join(self.exp.logs_dir, f"Scan_{scan_num}.log")

    def master_file(self, scan_num):
        return os.path.join(self.exp.scan_frames_dir, f"Scan_{scan_num}/Scan_{scan_num}_master.h5")
        
@dataclass
class ROI:

    scan: Scan 
    kind: str # Bragg or Pupil
    
    coords: np.ndarray = field(default=None, repr=False)

    # acquired later
    data_4d: np.ndarray = field(default=None, repr=False) # 4D Data set (x, y, fx, fy)
    centre_pixel: np.ndarray = field(default=None, repr=False)
    
    # to be set automatically
    __averaged_det_images: np.ndarray = field(default=None, repr=False)  # Average of all detector frames 
    __averaged_coherent_images: np.ndarray = field(default=None, repr=False)
    
    # Data
    # needs to be set manually
    detected_objects: np.ndarray = field(default=None, repr=False) # Detected Objects
    coherent_imgs: np.ndarray = field(default=None, repr=False)
    
    # k-space params
    kout_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kouts: np.ndarray = field(default=None, repr=False)  # Kout vectors
    kin_coords: np.ndarray = field(default=None, repr=False)  # Coords
    kins: np.ndarray = field(default=None, repr=False)  # Kin vectors
    
    kins_avg: np.ndarray = field(default=None, repr=False)  # average Kins 
    # For computing kins 
    optimal_angles: list = field(default=None, repr=False)  # Euler angles that rotate Kout to Kin
    est_ttheta: float = field(default=None, repr=False)
    g_init: np.ndarray = field(default=None, repr=False)  # Initial G vector for a Signal
        
    log_params: list = field(default_factory=list, repr = False)
    
    checkpoint_stack:list = field(default_factory=list, repr = False)


    # _______________ functionality with Exp ___________

    def __post_init__(self):
        # Copy attributes from Scan into ROI
        for f in vars(self.scan):
            setattr(self, f, getattr(self.scan, f))

    # _______________ Properties ________________
    @property
    def averaged_det_images(self):
        
        if self.__averaged_det_images is None:
            self.__averaged_det_images = np.mean(self.data_4d, axis=(0,1))
            
        return self.__averaged_det_images

    def update_averaged_det_images(self):
        self.__averaged_det_images = np.mean(self.data_4d, axis=(0,1))

    @averaged_det_images.setter
    def averaged_det_images(self, value):
        self.__averaged_det_images = value
        
    @property
    def averaged_coherent_images(self):
        
        if self.__averaged_coherent_images is None:
            self.__averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
        
        return self.__averaged_coherent_images
    
    @averaged_coherent_images.setter
    def averaged_coherent_images(self, value):
        self.__averaged_coherent_images = value
        
    def update_averaged_coherent_images(self):
        self.__averaged_coherent_images = np.mean(self.data_4d, axis=(2,3))
    
    
    # _______________ Saving ___________________
    
    @time_it
    def save(self, file_path):
    
        """
        Save the processed data and metadata to an HDF5 file.
    
        Args:

            file_path (str): The path where the HDF5 file will be saved.
        """

        with h5py.File(file_path, "w") as h5f:

            # Save metadata as attributes in the root group
            process_params = h5f.create_group("processing_params")
            
            process_params.attrs['roi'] = json.dumps(self.coords)
            process_params.attrs['step_size'] = self.step_size
            process_params.attrs['det_distance'] = self.det_distance
            
            images = h5f.create_group("processed_images")
            images.create_dataset("coherent_images", data=self.coherent_imgs, compression="gzip",chunks=True)
                
            kvectors = h5f.create_group("kvectors")
            kvectors.create_dataset("kins", data=self.kins, compression="gzip")
            kvectors.create_dataset("kouts", data = self.kouts, compression='gzip')
            kvectors.create_dataset("kin_coords", data=self.kin_coords, compression="gzip")
            kvectors.create_dataset("kout_coords", data = self.kout_coords, compression='gzip')
                
            print(f"Data saved at {file_path}")
    

    @classmethod
    def load_data(cls, file_path):
        """
        Load processed data and metadata for a specific ROI from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.
            file_path (str): Path to the HDF5 file containing saved data.
        """
        self = cls.__new__(cls)

        with h5py.File(file_path, "r") as h5f:
                                        
            # Load Processed Images 
            params = h5f["processing_params"]
            
            self.step_size = params.attrs['step_size']
            
            
            images = h5f["processed_images"]
            
            self.coherent_imgs = images["coherent_images"][...]
            
            # Load K-Vectors 
            kvectors = h5f["kvectors"]
            
            self.kins = kvectors['kins'][...]
            self.kouts = kvectors['kouts'][...]
            self.kin_coords = kvectors['kin_coords'][...]
            self.kout_coords = kvectors['kout_coords'][...]
                
        return self
    
    
    @time_it
    def save_4d(self, file_path):
    
        """
        Save the 4D data set for an roi to an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be saved.

            file_path (str): The path where the HDF5 file will be saved.
        """

        with h5py.File(file_path, "w") as h5f:
            data = h5f.create_group("data")
            # Save metadata as attributes in the root group
            data.create_dataset("data", data = self.data_4d, compression="gzip",chunks=True)

        print(f"Data saved at {file_path}")
    @time_it
    def load_4d(self, file_path):
    
        """
        Load the 4D data set for an roi from an HDF5 file.
    
        Args:
            roi_name (str): The ROI name for which the data should be loaded.

            file_path (str): The path where the HDF5 is stored.
        """
        print("Loading data ...")
        with h5py.File(file_path, "r") as h5f:
            data = h5f["data"]
            # Save metadata as attributes in the root group
            self.data_4d = np.array(data["data"])

        print("Data loaded")

    # _____________ Checkpointing _____________
    
    def checkpoint_state(self):
        """Save a deep copy of the current state for potential rollback."""
        state = copy.deepcopy({
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
            if key not in ("checkpoint_stack", "log_params")  # don't save logs or stack itself
        })
        self.checkpoint_stack.append(state)
        print(f"[ROI] Checkpoint #{len(self.checkpoint_stack)} created.")

    def restore_checkpoint(self):
        """Restore the most recent checkpoint."""
        if not self.checkpoint_stack:
            print("[ROI] No checkpoints to restore.")
            return

        last_state = self.checkpoint_stack.pop()
        for key, value in last_state.items():
            setattr(self, key, value)
    
    def auto_checkpoint(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.checkpoint_state()
            return func(self, *args, **kwargs)
        return wrapper
    



        