# Packages

import numpy as np
import cv2

from PIL import Image, ImageTk
from PIL.Image import fromarray

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *

# from os import listdir
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.figure import Figure

class ArrayVar(object):
    array_var = np.zeros((1024,1024),dtype=np.uint8)
    # def __init__(self, ):
    #     array_var = np.zeros((3,3,3))
    
    def set(self, value):
        self.array_var = value
        
    def get(self, ):
        return self.array_var

def bilinear_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  height, width = int(height), int(width)
  img_height, img_width = image.shape[0], image.shape[1]

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_lower = np.floor(x_ratio * x).astype('int32')
  y_lower = np.floor(y_ratio * y).astype('int32')

  x_upper = np.ceil(x_ratio * x).astype('int32')
  y_upper = np.ceil(y_ratio * y).astype('int32')

  x_weight = (x_ratio * x) - x_lower
  y_weight = (y_ratio * y) - y_lower

  traget_a = image[y_lower * img_width + x_lower]
  traget_b = image[y_lower * img_width + x_upper]
  traget_c = image[y_upper * img_width + x_lower]
  traget_d = image[y_upper * img_width + x_upper]

  traget = traget_a * (1 - x_weight) * (1 - y_weight) + \
           traget_b * x_weight * (1 - y_weight) + \
           traget_c * y_weight * (1 - x_weight) + \
           traget_d * x_weight * y_weight

  return traget.reshape(height, width)

def nearest_neighbor_resize_vectorized(image, height, width):
  """
  `image` is a 2-D numpy array
  `height` and `width` are the desired spatial dimension of the new 2-D array.
  """
  height, width = int(height), int(width)
  img_height, img_width = image.shape[0], image.shape[1]

  image = image.ravel()

  x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
  y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

  y, x = np.divmod(np.arange(height * width), width)

  x_c = np.round(x_ratio * x).astype('int32')
  y_c = np.round(y_ratio * y).astype('int32')
  traget = image[y_c * img_width + x_c]

  return traget.reshape(height, width)

gui = Tk()
gui.geometry("1200x800")
gui.title("ImageProcessTools")


file_path = StringVar()
image_array = ArrayVar()

new_height = DoubleVar()
new_width = DoubleVar()
new_power = DoubleVar()

resize_method = IntVar()

canvas = Canvas(gui, width=1024, height=1024)
canvas.pack()
# canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

init_img = fromarray(np.array(255 * np.ones((500,500)), dtype=np.uint8))
img_canvas = ImageTk.PhotoImage(init_img, master=gui)
image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=img_canvas)

def donothing():
    pass

def open_image():
    global new_img_canvas
    file_selected = filedialog.askopenfilename()
    file_selected = file_selected.lower()
    if 'bmp' in file_selected:
        image_array.set(cv2.imread(file_selected, cv2.IMREAD_GRAYSCALE))
    elif 'raw' in file_selected:
        image_array.set(np.fromfile(file_selected, dtype=np.uint8).reshape(512, 512))
    
    new_img_canvas = ImageTk.PhotoImage(fromarray(image_array.get()))
    canvas.itemconfigure(image_on_canvas, image=new_img_canvas)

    #canvas.update(img_canvas)
def save_image():
    file = filedialog.asksaveasfile(mode='wb', defaultextension=".bmp")
    if file:
        fromarray(image_array.get()).save(file)

def log_trans():
    global new_img_canvas
    image_array.set(np.array(50 * np.log(np.array(image_array.get(), dtype=np.int32) + 1), dtype=np.uint8))
    new_img_canvas = ImageTk.PhotoImage(fromarray(image_array.get()))
    canvas.itemconfigure(image_on_canvas, image=new_img_canvas)

# def gamma_trans(power=0.1):
#     global new_img_canvas
#     image_array.set(np.array(255 * ((np.array(image_array.get(), dtype=np.int32)/255) ** power), dtype=np.uint8))
#     new_img_canvas = ImageTk.PhotoImage(fromarray(image_array.get()))
#     canvas.itemconfigure(image_on_canvas, image=new_img_canvas)

def neg_trans():
    global new_img_canvas
    image_array.set(np.array(255 - np.array(image_array.get(), dtype=np.int32), dtype=np.uint8))
    new_img_canvas = ImageTk.PhotoImage(fromarray(image_array.get()))
    canvas.itemconfigure(image_on_canvas, image=new_img_canvas)

def gamma_trans():
    gamma_trans_new_wins = Toplevel(gui)
    gamma_trans_new_wins.title("gamma transform")
    gamma_trans_new_wins.geometry("150x100")
    
    power_label = Label(gamma_trans_new_wins, text ="Power: ")
    power_label.grid(row=0, column=0)
    
    power_entry = Entry(gamma_trans_new_wins, width=5)
    power_entry.grid(row=1,column=0)
    
    def get_power():
        global new_img_canvas
        
        new_power.set(float(power_entry.get()))
        image_array.set(np.array(255 * ((np.array(image_array.get(), dtype=np.int32)/255) ** new_power.get()), dtype=np.uint8))
        new_img_canvas = ImageTk.PhotoImage(fromarray(image_array.get()))
        canvas.itemconfigure(image_on_canvas, image=new_img_canvas)
        gamma_trans_new_wins.destroy()
    
    get_power_btn = tk.Button(gamma_trans_new_wins, text="Ok", command=get_power)
    get_power_btn.grid(row=2,column=0)

def resize_trans():
     
    # Toplevel object which will
    # be treated as a new window
    resize_trans_new_wins = Toplevel(gui)
 
    # sets the title of the
    # Toplevel widget
    resize_trans_new_wins.title("Resize")
 
    # sets the geometry of toplevel
    resize_trans_new_wins.geometry("200x180")
 
    # A Label widget to show in toplevel
    height_label = Label(resize_trans_new_wins, text ="Height: ")
    height_label.grid(row=1, column=0)
    
    height_entry = Entry(resize_trans_new_wins, width=5)
    height_entry.grid(row=1,column=1)
    
    width_label = Label(resize_trans_new_wins, text ="Width: ")
    width_label.grid(row=2, column=0)
    
    width_entry = Entry(resize_trans_new_wins, width=5)
    width_entry.grid(row=2,column=1)
    
    rdio_bilinear = tk.Radiobutton(resize_trans_new_wins, text='Bilinear', variable=resize_method, value=1)
    rdio_neighbor = tk.Radiobutton(resize_trans_new_wins, text='Neighbor', variable=resize_method, value=2)
    
    rdio_bilinear.grid(row=3,column=0)
    rdio_neighbor.grid(row=4,column=0)
    
    def get_height_and_width():
        global new_img_canvas
        
        new_height.set(float(height_entry.get()))
        new_width.set(float(width_entry.get()))
        if int(resize_method.get()) == 1:
            new_resize_img = bilinear_resize_vectorized(image_array.get(), new_height.get(), new_width.get())
        elif int(resize_method.get()) == 2:
            new_resize_img = nearest_neighbor_resize_vectorized(image_array.get(), new_height.get(), new_width.get())
        
        new_img_canvas = ImageTk.PhotoImage(fromarray(new_resize_img))
        canvas.itemconfigure(image_on_canvas, image=new_img_canvas)
        
        resize_trans_new_wins.destroy()
    
    get_h_and_w_btn = tk.Button(resize_trans_new_wins, text="Ok", command=get_height_and_width)
    get_h_and_w_btn.grid(row=5,column=1)
    
menubar = Menu(gui)
file_menu = Menu(menubar, tearoff=0)
file_menu.add_command(label="Open Image", command=open_image)
file_menu.add_command(label="Save Image", command=save_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=gui.destroy)
menubar.add_cascade(label="File", menu=file_menu)

opr_menu = Menu(menubar, tearoff=0)
opr_menu.add_command(label="log", command=log_trans)
opr_menu.add_command(label="gamma", command=gamma_trans)
opr_menu.add_command(label="negative", command=neg_trans)
opr_menu.add_separator()
opr_menu.add_command(label="resize", command=resize_trans)
menubar.add_cascade(label="Operation", menu=opr_menu)

gui.config(menu=menubar)

gui.mainloop()
