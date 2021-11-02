
import numpy as np
import cv2

from PIL import Image, ImageTk
from PIL.Image import fromarray

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *


# import os
# os.chdir(r"C:\Users\Taner\Documents\GitHub\DIPCV\assignment2")

from utils.Trans import Transform
from utils.Conv_Filter import Conv_Filter
from utils.Hist_Enhencement import Hist_Enhencement

class ArrayVar(object):
    array_var = np.zeros((1024, 1024),dtype=np.uint8)
    
    def set(self, value):
        self.array_var = value
        
    def get(self, ):
        return self.array_var

class MainApplication(Tk, Transform, Conv_Filter, Hist_Enhencement):
    def __init__(self):
        Tk.__init__(self)
        Transform.__init__(self)
        Conv_Filter.__init__(self)
        Hist_Enhencement.__init__(self)
        
        # set windows
        self.title("ImageProcessTools")   	 
        self.geometry("1200x800")
        self.init_menubar()
        
        # variable
        self.image_array = ArrayVar()
        self.resize_method = IntVar()
        
        # canvas
        self.canvas = Canvas(self, width=1024, height=1024)
        self.canvas.pack()
        
        self.init_img = fromarray(np.array(255 * np.ones((500,500)), dtype=np.uint8))
        self.img_canvas = ImageTk.PhotoImage(self.init_img, master=self)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img_canvas)
        
    def init_menubar(self):
        menubar = Menu(self)
        
        FileMenu = Menu(menubar, tearoff=0)
        OprMenu = Menu(menubar, tearoff=0)
        FilterMenu = Menu(menubar, tearoff=0)
        
        menubar.add_cascade(label="File", menu=FileMenu)
        menubar.add_cascade(label="Operation", menu=OprMenu)
        menubar.add_cascade(label="Filter", menu=FilterMenu)
        
        FileMenu.add_command(label="Open Image", command=self.open_image)
        FileMenu.add_command(label="Save Image", command=self.save_image)
        FileMenu.add_separator()
        FileMenu.add_command(label="Exit", command=self.destroy)
        
        OprMenu.add_command(label="Log", command=self.log_opr)
        OprMenu.add_command(label="Gamma", command=self.gamma_opr)
        OprMenu.add_command(label="Negative", command=self.neg_opr)
        OprMenu.add_separator()
        OprMenu.add_command(label="Resize", command=self.resize_opr)
        
        FilterMenu.add_command(label="Global Hist", command=self.global_hist_eq)
        FilterMenu.add_command(label="Local Hist")
        FilterMenu.add_command(label="Hist match")
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Gaussian")
        FilterMenu.add_command(label="Averaging")
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Unsharp")
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Laplacian")
        FilterMenu.add_command(label="Sobel")
        
        self.configure(menu=menubar)
    
    def global_hist_eq(self):
        self.image_array.set(self.HistEq(self.image_array.get()))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
    
    def local_hist_eq(self):
        pass
        
    def open_image(self):
        # global new_img_canvas
        file_selected = filedialog.askopenfilename()
        file_selected = file_selected.lower()
        
        if 'bmp' in file_selected:
            self.image_array.set(cv2.imread(file_selected, cv2.IMREAD_GRAYSCALE))
        elif 'raw' in file_selected:
            self.image_array.set(np.fromfile(file_selected, dtype=np.uint8).reshape(512, 512))
        
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)

        #canvas.update(img_canvas)
    def save_image(self):
        file = filedialog.asksaveasfile(mode='wb', defaultextension=".bmp")
        if file:
            fromarray(self.image_array.get()).save(file)
            
    def log_opr(self):
        self.image_array.set(self.log_trans(self.image_array.get()))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
    
    def gamma_opr(self):
        gamma_trans_new_wins = Toplevel(self)
        gamma_trans_new_wins.title("gamma transform")
        gamma_trans_new_wins.geometry("150x100")
        
        power_label = Label(gamma_trans_new_wins, text ="Power: ")
        power_label.grid(row=0, column=0)
        
        power_entry = Entry(gamma_trans_new_wins, width=5)
        power_entry.grid(row=1,column=0)
        
        def get_power():
            g_power = float(power_entry.get())
            self.image_array.set(self.gamma_trans(self.image_array.get(), g_power))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            gamma_trans_new_wins.destroy()
        
        get_power_btn = tk.Button(gamma_trans_new_wins, text="Ok", command=get_power)
        get_power_btn.grid(row=2,column=0)
    
    def neg_opr(self):
        self.image_array.set(self.neg_trans(self.image_array.get()))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
        
    def resize_opr(self):
        resize_trans_new_wins = Toplevel(self)
        resize_trans_new_wins.title("Resize")
        resize_trans_new_wins.geometry("200x180")
        
        height_label = Label(resize_trans_new_wins, text ="Height: ")
        height_label.grid(row=1, column=0)
        
        height_entry = Entry(resize_trans_new_wins, width=5)
        height_entry.grid(row=1,column=1)
        
        width_label = Label(resize_trans_new_wins, text ="Width: ")
        width_label.grid(row=2, column=0)
        
        width_entry = Entry(resize_trans_new_wins, width=5)
        width_entry.grid(row=2,column=1)
        
        rdio_bilinear = tk.Radiobutton(resize_trans_new_wins, text='Bilinear', variable=self.resize_method, value=1)
        rdio_neighbor = tk.Radiobutton(resize_trans_new_wins, text='Neighbor', variable=self.resize_method, value=2)
        
        rdio_bilinear.grid(row=3,column=0)
        rdio_neighbor.grid(row=4,column=0)
        
        def get_height_and_width():
            
            new_height = int(height_entry.get())
            new_width = int(width_entry.get())
            
            new_resize_img = self.image_resize(self.image_array.get(), new_height, new_width, mode=int(self.resize_method.get()))
            self.image_array.set(new_resize_img)
            
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            
            resize_trans_new_wins.destroy()
        
        get_h_and_w_btn = tk.Button(resize_trans_new_wins, text="Ok", command=get_height_and_width)
        get_h_and_w_btn.grid(row=5,column=1)
            
app=MainApplication()
app.mainloop()
