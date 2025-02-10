import h5py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
import numpy as np

class HDF5Viewer:
    def __init__(self, root):
        self.root = root
        self.root.title("HDF5 File Viewer")

        # File selection button
        self.file_label = tk.Label(root, text="No file selected")
        self.file_label.pack(pady=5)
        
        self.open_button = tk.Button(root, text="Open HDF5 File", command=self.load_file)
        self.open_button.pack(pady=5)

        # Treeview to display file structure
        self.tree = ttk.Treeview(root)
        self.tree.pack(padx=10, pady=10, expand=True, fill="both")
        self.tree.bind("<Double-1>", self.on_tree_select)

        # Metadata and dataset details
        self.info_text = tk.Text(root, height=10, wrap="word")
        self.info_text.pack(padx=10, pady=5, expand=True, fill="both")

        self.view_button = tk.Button(root, text="View Dataset", command=self.view_dataset)
        self.view_button.pack(pady=5)

        self.file_path = None
        self.selected_item = None

    def load_file(self):
        """Open and load an HDF5 file."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select HDF5 File",
                filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
            )
            if not file_path:
                return
            self.file_path = file_path
            self.file_label.config(text=f"Loaded: {file_path}")
            self.display_structure()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def display_structure(self):
        """Display HDF5 file structure in the tree view."""
        if not self.file_path:
            return
        
        self.tree.delete(*self.tree.get_children())  # Clear previous contents
        
        try:
            with h5py.File(self.file_path, "r") as h5f:
                def insert_nodes(parent, h5group):
                    for name, item in h5group.items():
                        node = self.tree.insert(parent, "end", text=name, open=False)
                        if isinstance(item, h5py.Group):
                            insert_nodes(node, item)
                        elif isinstance(item, h5py.Dataset):
                            self.tree.insert(node, "end", text=f"{name} (Dataset)")
                
                insert_nodes("", h5f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read file: {e}")

    def on_tree_select(self, event):
        """Show dataset details or metadata when a node is selected."""
        selected_item = self.tree.selection()
        if not selected_item:
            return
        
        self.selected_item = self.tree.item(selected_item)["text"]
        self.info_text.delete("1.0", tk.END)
        
        try:
            with h5py.File(self.file_path, "r") as h5f:
                def get_item_by_path(h5group, path_list):
                    """Recursively navigate to the selected item."""
                    if not path_list:
                        return h5group
                    return get_item_by_path(h5group[path_list[0]], path_list[1:])
                
                path_list = self.selected_item.split("/")
                item = get_item_by_path(h5f, path_list)

                if isinstance(item, h5py.Dataset):
                    self.info_text.insert(tk.END, f"Dataset: {self.selected_item}\n")
                    self.info_text.insert(tk.END, f"Shape: {item.shape}\n")
                    self.info_text.insert(tk.END, f"Data Type: {item.dtype}\n")
                    
                    # Show first few values
                    preview = item[:5] if item.ndim == 1 else item[:5, :5]
                    self.info_text.insert(tk.END, f"Preview:\n{preview}\n")

                elif isinstance(item, h5py.Group):
                    self.info_text.insert(tk.END, f"Group: {self.selected_item}\n")
                
                # Display attributes
                if item.attrs:
                    self.info_text.insert(tk.END, "\nAttributes:\n")
                    for key, value in item.attrs.items():
                        self.info_text.insert(tk.END, f"{key}: {value}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read item: {e}")

    def view_dataset(self):
        """Visualize a dataset if it's 2D."""
        if not self.selected_item or not self.file_path:
            messagebox.showerror("Error", "No dataset selected!")
            return

        try:
            with h5py.File(self.file_path, "r") as h5f:
                dataset = h5f[self.selected_item]
                if dataset.ndim == 2:  # Only show 2D datasets
                    plt.figure(figsize=(6, 6))
                    plt.imshow(dataset, cmap="gray")
                    plt.colorbar(label="Intensity")
                    plt.title(self.selected_item)
                    plt.show()
                else:
                    messagebox.showwarning("Warning", "Only 2D datasets can be visualized.")
        except KeyError:
            messagebox.showerror("Error", "Dataset not found!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize dataset: {e}")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HDF5Viewer(root)
    root.mainloop()