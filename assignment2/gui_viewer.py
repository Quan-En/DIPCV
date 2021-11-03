
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
        self.sobel_direction = IntVar()
        self.special_kernel = IntVar()
        self.use_smooth = IntVar()
        
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
        FilterMenu.add_command(label="Local Hist", command=self.local_hist_eq)
        FilterMenu.add_command(label="Hist match", command=self.global_hist_match)
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Gaussian", command=self.gaussian_blur)
        FilterMenu.add_command(label="Averaging", command=self.average_blur)
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Unsharp", command=self.unsharp_edge_enhence)
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Laplacian", command=self.laplacian_edge_detect)
        FilterMenu.add_command(label="Sobel", command=self.sobel_edge_detect)
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Median", command=self.median_denoise)
        FilterMenu.add_command(label="Bilateral", command=self.bilateral_denoise)
        FilterMenu.add_command(label="Special", command=self.special_deal)
        FilterMenu.add_separator()
        FilterMenu.add_command(label="Non local means(NLM)", command=self.nl_means)
        
        self.configure(menu=menubar)

    def nl_means(self):
        new_wins = Toplevel(self)
        new_wins.title("NLM")
        new_wins.geometry("240x180")
        
        win_size1_label = Label(new_wins, text ="kernel size: ")
        win_size1_label.grid(row=0, column=0)
        
        size1_entry = Entry(new_wins, width=5)
        size1_entry.grid(row=0, column=1)
        
        win_size2_label = Label(new_wins, text ="search size: ")
        win_size2_label.grid(row=1, column=0)
        
        size2_entry = Entry(new_wins, width=5)
        size2_entry.grid(row=1, column=1)

        h_label = Label(new_wins, text ="h: ")
        h_label.grid(row=2, column=0)
        
        h_entry = Entry(new_wins, width=5)
        h_entry.grid(row=2, column=1)
        
        def get_parm():
            g_size1 = int(size1_entry.get())
            g_size2 = int(size2_entry.get())
            g_h = float(h_entry.get())
            self.image_array.set(self.nonLocalMeans_filter(self.image_array.get(), g_size1, g_size2, g_h))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
            
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=4,column=0)
        

    def global_hist_match(self):
        file_selected = filedialog.askopenfilename()
        file_selected = file_selected.lower()
        
        if 'bmp' in file_selected:
            temp = cv2.imread(file_selected, cv2.IMREAD_GRAYSCALE)
        elif 'raw' in file_selected:
            temp = np.fromfile(file_selected, dtype=np.uint8).reshape(512, 512)
            
        self.image_array.set(self.hist_match(self.image_array.get(), temp))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
    

    
    def bilateral_denoise(self):
        new_wins = Toplevel(self)
        new_wins.title("Bilateral denoise")
        new_wins.geometry("240x180")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0, column=1)
        
        sigma_c_label = Label(new_wins, text ="Sigma(Distance): ")
        sigma_c_label.grid(row=1, column=0)
        
        sigma_c_entry = Entry(new_wins, width=5)
        sigma_c_entry.grid(row=1, column=1)
        
        sigma_s_label = Label(new_wins, text ="Sigma(Intensities): ")
        sigma_s_label.grid(row=2, column=0)
        
        sigma_s_entry = Entry(new_wins, width=5)
        sigma_s_entry.grid(row=2, column=1)
        
        rdio_1 = tk.Radiobutton(new_wins, text='yes', variable=self.use_smooth, value=1)
        rdio_2 = tk.Radiobutton(new_wins, text='no', variable=self.use_smooth, value=0)
        
        rdio_1.grid(row=3,column=0)
        rdio_2.grid(row=3,column=1)
        
        def get_parm():
            g_size = int(size_entry.get())
            g_use_smooth = int(self.use_smooth.get())
            if g_use_smooth:
                g_sigma_c = float(sigma_c_entry.get())
            else:
                g_sigma_c = 1
            g_sigma_s = float(sigma_s_entry.get())
            
            self.image_array.set(self.bilateral_filter(self.image_array.get(), g_size, g_sigma_c, g_sigma_s, g_use_smooth))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
            
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=4,column=0)
    
    def median_denoise(self):
        new_wins = Toplevel(self)
        new_wins.title("Special kernel")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0,column=1)
        
        def get_parm():
            g_size = int(size_entry.get())
            self.image_array.set(self.denoising(self.image_array.get(), g_size, 'median'))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=3,column=0)
    
    def special_deal(self):
        
        new_wins = Toplevel(self)
        new_wins.title("Special kernel")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0,column=1)
        
        rdio_1 = tk.Radiobutton(new_wins, text='kernel-1', variable=self.special_kernel, value=1)
        rdio_2 = tk.Radiobutton(new_wins, text='kernel-2', variable=self.special_kernel, value=2)
        
        rdio_1.grid(row=2,column=0)
        rdio_2.grid(row=3,column=0)
        
        def get_parm():
            g_size = int(size_entry.get())
            g_index = int(self.special_kernel.get())
            self.image_array.set(self.special_filter(self.image_array.get(), g_size, 'gaussian', g_index))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=4,column=0)
    
    def unsharp_edge_enhence(self):
        new_wins = Toplevel(self)
        new_wins.title("Unsharp edge enhence")
        new_wins.geometry("180x150")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0,column=1)
        
        sigma_label = Label(new_wins, text ="sigma: ")
        sigma_label.grid(row=1, column=0)
        
        sigma_entry = Entry(new_wins, width=5)
        sigma_entry.grid(row=1,column=1)
        
        amount_label = Label(new_wins, text ="times amount: ")
        amount_label.grid(row=2, column=0)
        
        amount_entry = Entry(new_wins, width=5)
        amount_entry.grid(row=2,column=1)
        
        def get_parm():
            g_size = int(size_entry.get())
            g_sigma = float(sigma_entry.get())
            g_amount = float(amount_entry.get())
            self.image_array.set(self.unsharp(self.image_array.get(), g_size, 'gaussian', g_amount, sigma=g_sigma))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=3,column=0)
    
    def global_hist_eq(self):
        self.image_array.set(self.HistEq(self.image_array.get()))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
    
    def local_hist_eq(self):
        new_wins = Toplevel(self)
        new_wins.title("Local Histogram Equalization")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=1,column=0)
        
        def get_parm():
            g_size = int(size_entry.get())
            self.image_array.set(self.Local_HistEq(self.image_array.get(), g_size))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=2,column=0)
    
    def gaussian_blur(self):
        new_wins = Toplevel(self)
        new_wins.title("Gaussian blur")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0,column=1)
        
        sigma_label = Label(new_wins, text ="sigma: ")
        sigma_label.grid(row=1, column=0)
        
        sigma_entry = Entry(new_wins, width=5)
        sigma_entry.grid(row=1,column=1)
        
        def get_parm():
            g_size = int(size_entry.get())
            g_sigma = float(sigma_entry.get())
            self.image_array.set(self.blur(self.image_array.get(), g_size, 'gaussian', sigma=g_sigma))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=2,column=0)
        
    def average_blur(self):
        new_wins = Toplevel(self)
        new_wins.title("Averaging blur")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=1,column=0)
        
        def get_parm():
            g_size = int(size_entry.get())
            self.image_array.set(self.blur(self.image_array.get(), g_size, 'average'))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=2,column=0)
    
    def laplacian_edge_detect(self):
        new_wins = Toplevel(self)
        new_wins.title("Laplacian edge detect")
        new_wins.geometry("150x100")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=1,column=0)
        
        def get_parm():
            g_size = int(size_entry.get())
            self.image_array.set(self.laplacian_filter(self.image_array.get(), g_size))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=2,column=0)
        
    def sobel_edge_detect(self):
        new_wins = Toplevel(self)
        new_wins.title("Sobel edge detect")
        new_wins.geometry("180x150")
        
        win_size_label = Label(new_wins, text ="kernel size: ")
        win_size_label.grid(row=0, column=0)
        
        size_entry = Entry(new_wins, width=5)
        size_entry.grid(row=0,column=1)
        
        rdio_row = tk.Radiobutton(new_wins, text='By row', variable=self.sobel_direction, value=1)
        rdio_col = tk.Radiobutton(new_wins, text='By column', variable=self.sobel_direction, value=2)
        
        rdio_row.grid(row=2,column=0)
        rdio_col.grid(row=3,column=0)
        
        def get_parm():
            g_size = int(size_entry.get())
            g_dir = int(self.sobel_direction.get())
            self.image_array.set(self.sobel_filter(self.image_array.get(), g_size, g_dir))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=4,column=0)
        
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
        new_wins = Toplevel(self)
        new_wins.title("gamma transform")
        new_wins.geometry("150x100")
        
        power_label = Label(new_wins, text ="Power: ")
        power_label.grid(row=0, column=0)
        
        power_entry = Entry(new_wins, width=5)
        power_entry.grid(row=1,column=0)
        
        def get_parm():
            g_power = float(power_entry.get())
            self.image_array.set(self.gamma_trans(self.image_array.get(), g_power))
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=2,column=0)
    
    def neg_opr(self):
        self.image_array.set(self.neg_trans(self.image_array.get()))
        self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
        self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
        
    def resize_opr(self):
        new_wins = Toplevel(self)
        new_wins.title("Resize")
        new_wins.geometry("200x180")
        
        height_label = Label(new_wins, text ="Height: ")
        height_label.grid(row=1, column=0)
        
        height_entry = Entry(new_wins, width=5)
        height_entry.grid(row=1,column=1)
        
        width_label = Label(new_wins, text ="Width: ")
        width_label.grid(row=2, column=0)
        
        width_entry = Entry(new_wins, width=5)
        width_entry.grid(row=2,column=1)
        
        rdio_bilinear = tk.Radiobutton(new_wins, text='Bilinear', variable=self.resize_method, value=1)
        rdio_neighbor = tk.Radiobutton(new_wins, text='Neighbor', variable=self.resize_method, value=2)
        
        rdio_bilinear.grid(row=3,column=0)
        rdio_neighbor.grid(row=4,column=0)
        
        def get_parm():
            
            new_height = int(height_entry.get())
            new_width = int(width_entry.get())
            new_mode = int(self.resize_method.get())
            new_resize_img = self.image_resize(self.image_array.get(), new_height, new_width, mode=new_mode)
            self.image_array.set(new_resize_img)
            
            self.new_img_canvas = ImageTk.PhotoImage(fromarray(self.image_array.get()))
            self.canvas.itemconfigure(self.image_on_canvas, image=self.new_img_canvas)
            
            new_wins.destroy()
        
        get_btn = tk.Button(new_wins, text="Ok", command=get_parm)
        get_btn.grid(row=5,column=1)
            
app=MainApplication()
app.mainloop()
