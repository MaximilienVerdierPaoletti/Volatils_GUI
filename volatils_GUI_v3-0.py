#!/usr/bin/env python



#---------------------------- Version 1.0

# Added flexibility on the names of standard. Miswritten names can still be identified.
# Added flexibility on the numbers of elements analyzed. Would work for one, two, three, four ratios
# Added flexibility on the identification of elements. Now Fluor can be interpreted as well.



#---------------------------- Version 2.0

# Added auto adjustment of window size to screen resolution

import tkinter
from tkinter import filedialog, messagebox, ttk
import customtkinter
from PIL import Image, ImageTk
import os
import pandas as pd


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


#=================== Necessary modules for linear regressions
import numpy as np
import matplotlib
import matplotlib.pyplot as plt # Enables plotting of data
from matplotlib.patches import ConnectionPatch
import pylab # Enables the use of the 'savefig' function
import math
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import t #import of t-table for confidence interval calculation
from sklearn.preprocessing import PolynomialFeatures
import scipy
from scipy.optimize import curve_fit
from lmfit import Model
import difflib
import re
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2)
# from win32api import GetSystemMetrics



PATH = os.path.dirname(os.path.realpath(__file__))

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTkToplevel):



    # screen_width = GetSystemMetrics(0)
    # screen_height = GetSystemMetrics(1)

    screen_width = customtkinter.CTk().winfo_screenwidth()
    screen_height = customtkinter.CTk().winfo_screenheight()

    RATIO = screen_width/screen_height
    WIDTH = int(np.round(screen_width*0.70))
    HEIGHT = int(WIDTH/RATIO)
    screen_resolution=str(WIDTH)+'x'+str(HEIGHT)
    print(screen_height,screen_width,HEIGHT,WIDTH)

    def __init__(self):
        super().__init__()

        self.title("Volatils Explorer [V2.0]")
        # self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.wm_minsize(width=App.WIDTH,height=App.HEIGHT)
        # self.geometry(App.screen_resolution)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        #### Load images for buttons
        self.add_folder_image = customtkinter.CTkImage(light_image=Image.open("./test_images/add-folder.png"),
                                  dark_image=Image.open("./test_images/add-folder.png"),
                                  size=(20, 20))
        self.plot_image = customtkinter.CTkImage(light_image=Image.open("./test_images/plot.png"),
                                  dark_image=Image.open("./test_images/plot.png"),
                                  size=(30, 30))
        self.regplot_image = customtkinter.CTkImage(light_image=Image.open("./test_images/reg_plot.png"),
                                  dark_image=Image.open("./test_images/reg_plot.png"),
                                  size=(30, 30))

        data=None
        data_corr=None
        stand_meas=None
        text_stat=None
        Er_comp=None

        #### ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=int(App.WIDTH*0.2),
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        #### ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        # self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        # self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        # self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Volatils Explorer",
                                              font=("Roboto Medium", -16))  # font name and size in px
        self.label_1.grid(row=1, column=0, pady=10, padx=10)

        self.button_File = customtkinter.CTkButton(master=self.frame_left, image=self.add_folder_image, text="Select file", height=32,
                                                compound="right", command=self.Open_File)
        self.button_File.grid(row=2, column=0, pady=10, padx=20)


        self.button_plot = customtkinter.CTkButton(master=self.frame_left, text="Plot", image=self.plot_image, height=32,
                                                compound="right",fg_color="#D35B58", hover_color="#C77C78", command=self.Plot_Data)
        self.button_plot.grid(row=3, column=0, pady=10, padx=20)



        self.button_regplot = customtkinter.CTkButton(master=self.frame_left,image=self.regplot_image, text="Regression", height=32,
                                                compound="right",fg_color="#3EC97D", hover_color="#59E397", command=self.Reg_Plot)
        self.button_regplot.grid(row=4, column=0, pady=10, padx=20)


##        self.button_testlogic = customtkinter.CTkButton(master=self.frame_left, image=self.add_list_image, text="Test Logic", height=32,
##                                                compound="right", fg_color="#D35B58", hover_color="#C77C78",
##                                                command=self.test_logic)
##        self.button_testlogic.grid(row=4, column=0, columnspan=2, padx=20, pady=10)


        self.filename_frame=customtkinter.CTkFrame(master=self.frame_left)
        self.filename_frame.grid(row=5, column=0, sticky="we",pady=20, padx=20,)
        self.label_filename = customtkinter.CTkLabel(master=self.filename_frame, text="File Name:", corner_radius=10)
        self.label_filename.grid(row=0, column=0, pady=0, padx=0, sticky="w")

        self.label_mode = customtkinter.CTkLabel(master=self.frame_left, text="Appearance Mode:")
        self.label_mode.grid(row=11, column=0, pady=20, padx=20, sticky="w")

        self.optionmenu_1 = customtkinter.CTkOptionMenu(master=self.frame_left,
                                                        values=["Light", "Dark", "System"],
                                                        command=self.change_appearance_mode)
        self.optionmenu_1.grid(row=12, column=0, pady=10, padx=20, sticky="w")

        #### ============ frame_right ============

        # configure grid layout (3x5)
        self.frame_right.rowconfigure(0, weight=6)
        self.frame_right.rowconfigure((1, 2), weight=1)
        self.frame_right.columnconfigure((0, 1, 3, 4, 5), weight=1)
        # self.frame_right.columnconfigure(2, weight=0)


        #### ============ Treeview ============

        self.frame_table = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_table.grid(row=0, column=3, columnspan=2, pady=20, padx=20, sticky="nsew")
        # configure grid layout (1x1)
        self.frame_table.rowconfigure(0, weight=1)
        self.frame_table.columnconfigure(0, weight=1)
        self.frame_table.pack_propagate(0)  # Forbids the frame size to be adjusted to the table size

        self.tv1=ttk.Treeview(master=self.frame_table)#self.frame_table.cget("height")
        ttk.Style(self.tv1).configure('Treeview',rowheight=40)
        self.tv1.place(relheight=1,relwidth=1)
        treescrolly=customtkinter.CTkScrollbar(master=self.frame_table, orientation="vertical", command=self.tv1.yview)
        treescrollx=customtkinter.CTkScrollbar(master=self.frame_table, orientation="horizontal", command=self.tv1.xview)
        self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
        treescrollx.pack(side="bottom", fill="x")
        treescrolly.pack(side="right", fill="y")
        self.tv1.pack(fill='both',expand=True)

        ####============ Plot Window =============

        self.plot_window = customtkinter.CTkFrame(master=self.frame_right)
        self.plot_window.grid(row=0, column=0, columnspan=3, pady=20, padx=20, sticky="nsew")
        self.plot_window.pack_propagate(0)
##        self.label_plot_window = customtkinter.CTkLabel(master=self.frame_right,
##                                                        text="Calibrations",
##                                                        fg_color=("white", "gray75"),
##                                                        corner_radius=8, width=120,font=("Roboto Medium",10))
##        self.label_plot_window.grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="")


        #### ============ frame_right buttons ============


        self.frame_buttonsCI = customtkinter.CTkFrame(master=self.frame_left,width=100,height=100)
        self.frame_buttonsCI.grid(row=6, column=0, pady=10, padx=10, sticky="nsew")
        self.frame_buttonsCI.rowconfigure(3,weight=1)

        self.buttonsCI_var = tkinter.IntVar(value=0)

        self.label_radio_group = customtkinter.CTkLabel(master=self.frame_buttonsCI,
                                                        text="Regression Error \n Propagation Parameters:")
        self.label_radio_group.grid(row=0, column=0, pady=10, padx=10, sticky="")

        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.frame_buttonsCI,
                                                           variable=self.buttonsCI_var,
                                                           value=0,text="Confidence Interval",
                                                           radiobutton_width=15,
                                                           radiobutton_height=15)
        self.radio_button_1.grid(row=1, column=0, pady=10, padx=10, sticky="w")

        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.frame_buttonsCI,
                                                           variable=self.buttonsCI_var,
                                                           value=1,text="Prediction Interval",
                                                           radiobutton_width=15,
                                                           radiobutton_height=15)
        self.radio_button_2.grid(row=2, column=0, pady=10, padx=10, sticky="w")








        self.frame_precinterval = customtkinter.CTkFrame(master=self.frame_left,width=100,height=100)
        self.frame_precinterval.grid(row=7, column=0, pady=10, padx=10, sticky="nsew")
        self.frame_precinterval.rowconfigure(3,weight=1)
        self.frame_precinterval.columnconfigure(1,weight=1)




        self.precinterval_var = tkinter.IntVar(value=0)
        self.label_radio_precinterval = customtkinter.CTkLabel(master=self.frame_precinterval,
                                                        text="Confidence interval \n percentage:")
        self.label_radio_precinterval.grid(row=0, column=0, pady=10, padx=10, sticky="")


        self.precinterval_button_1 = customtkinter.CTkRadioButton(master=self.frame_precinterval,
                                                           variable=self.precinterval_var,
                                                           value=0,text="95%",
                                                           radiobutton_width=15,
                                                           radiobutton_height=15)
        self.precinterval_button_1.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

        self.precinterval_button_2 = customtkinter.CTkRadioButton(master=self.frame_precinterval,
                                                           variable=self.precinterval_var,
                                                           value=1,text="99%",
                                                           radiobutton_width=15,
                                                           radiobutton_height=15)
        self.precinterval_button_2.grid(row=2, column=0, pady=10, padx=10, sticky="ew")




        self.check_var_error = customtkinter.StringVar(value="off")
        self.check_box_error = customtkinter.CTkCheckBox(master=self.frame_right,
                                                     text="Error Recap",variable=self.check_var_error,onvalue="on", offvalue="off",command=self.error_recap)
        self.check_box_error.grid(row=2, column=0, pady=5, padx=20, sticky="w")

        self.check_var = customtkinter.StringVar(value="off")
        self.check_box_testsummary = customtkinter.CTkCheckBox(master=self.frame_right,
                                                     text="Test summaries",variable=self.check_var,onvalue="on", offvalue="off",command=self.test_summaries)
        self.check_box_testsummary.grid(row=2, column=1, pady=5, padx=20, sticky="w")

        self.check_var_testlogic = customtkinter.StringVar(value="off")
        self.check_box_testlogic = customtkinter.CTkCheckBox(master=self.frame_right,
                                                     text="Test logic",variable=self.check_var_testlogic,onvalue="on", offvalue="off",command=self.test_logic)
        self.check_box_testlogic.grid(row=2, column=2, pady=5, padx=20, sticky="w")

        self.switch_var = customtkinter.StringVar(value="on")
        self.switch_1 = customtkinter.CTkSwitch(master=self.frame_right,
                                                text="Measurements",command=self.Switch_Event,
                                                variable=self.switch_var,onvalue="on", offvalue="off")
        self.switch_1.grid(row=2, column=4, pady=5, padx=20, sticky="w")


        self.entry = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,
                                            placeholder_text="Name of files")
        self.entry.grid(row=3, column=0, columnspan=4, pady=20, padx=20, sticky="we")

        self.button_5 = customtkinter.CTkButton(master=self.frame_right,
                                                text="Save",
                                                border_width=2,  # <- custom border_width
                                                fg_color=None,  # <- no fg_color
                                                command=self.Save_Files)
        self.button_5.grid(row=3, column=4, columnspan=1, pady=20, padx=20, sticky="we")

        #### set default values
        self.optionmenu_1.set("Dark")
        self.radio_button_1.select()
        self.precinterval_button_1.select()
        #self.progressbar.set(0.5)
        self.switch_1.select()
        self.check_box_testsummary.deselect()
        self.check_box_testlogic.deselect()
        self.check_box_error.deselect()


#**************************************************************************************************************#
#--------------------------------------------------------------------------------------------------------------#
#==============================================================================================================#
#                                                   FUNCTIONS                                                  #
#==============================================================================================================#
#--------------------------------------------------------------------------------------------------------------#
#**************************************************************************************************************#




#==============================================================================================================#
#======================================= File Opening Button Function =========================================#
#==============================================================================================================#


    def Open_File(self):
        global data, stand_meas, filename
        filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a file",
                                          filetype=(("xlsx files","*.xlsx"),("All Files","*.*")))
        #label_file["text"]=filename

        file_path= filename
        try:
            excel_filename=r"{}".format(file_path)
            multi_sheet_file=pd.ExcelFile(excel_filename)
            excel_sheet_names=multi_sheet_file.sheet_names
            dict_of_sheets= {}
            self.label_filename.configure(text="File Name: \n\n"+os.path.basename(filename), wraplength=100, justify = 'center')

            for sheet in excel_sheet_names:
                dict_of_sheets[sheet]=pd.read_excel(multi_sheet_file, sheet_name=sheet,header=0)
                if sheet == 'Std_Python':
                    stand_meas=pd.read_excel(multi_sheet_file, sheet_name=sheet,header=0)
                elif sheet == 'Meas_Python':
                    data=pd.read_excel(multi_sheet_file, sheet_name=sheet,header=0)

        except ValueError:
            tkinter.messagebox.showerror("Information","The file you have chosen is invalid")
            return None
        except FileNotFoundError:
            tkinter.messagebox.showerror("Information",f"No such file as {file_path}")



        return None


#==============================================================================================================#
#============================================ Plot Data Function ==============================================#
#==============================================================================================================#


    def Plot_Data(self):
        global data, stand_meas, filename

        #Clear old treeview
        self.clear_data()


        #---------- Extracting standards true concentrations
        stand_true=pd.read_excel('Standards_concentrations.xlsx',header=0)


        # stand_meas=pd.read_excel(name_datafile+'.xlsx',sheet_name='Std_Python',header=0)
        l=stand_meas.columns.to_list()
        stand=stand_meas[[m for m in l if 'Ratio' in m or 'Err Mean' in m]]
        stand.insert(0,"Name",stand_meas.Filename)


        del l

        ls_stand=stand_true.NAME.tolist()
        d=pd.DataFrame(columns=stand_true.columns.to_list())
        for i in range(0,len(stand)):
            X=stand.iloc[i]
            Xname=re.sub(r'|'.join(('chain\d+_\d+','@\d+_\d+')),'',X.Name,flags=re.IGNORECASE)
            print(Xname)
            l=difflib.get_close_matches(Xname,ls_stand,n=1,cutoff=0) # Retrieve good name of standards (even if variations are present)
            # print(Xname, l)
            d=pd.concat([d, stand_true.loc[stand_true.NAME==l[0]]], ignore_index=True)

        stand=pd.concat([stand,d],axis=1)

        del l, X, d

        generalcolor='k'
        FT=12
        FT_L=10
        # FC=generalcolor
        # white=np.array([1,1,1,-0.25])

        plt.rcParams['axes.linewidth'] = 2 # Width of borders
        #plt.rcParams['axes.facecolor'] = 'k'
        plt.rcParams['axes.edgecolor'] = generalcolor
        plt.rcParams['lines.color'] = generalcolor
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 2
        plt.rcParams['xtick.labelsize'] = FT_L
        plt.rcParams['xtick.color'] = generalcolor
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['ytick.minor.width'] = 2
        plt.rcParams['ytick.labelsize'] = FT_L
        plt.rcParams['ytick.color'] = generalcolor




        #### Ratios identification

        rat=[re.sub(' Ratio','',i) for i in stand.columns if ' Ratio' in i]
        T=[re.sub(r'|'.join(('/28Si','\d+')),'',i) for i in rat]
        T=list(map(lambda x: x.replace('O', 'OH'), T))
        C=['tomato','royalblue','gold','mediumseagreen']

        if len(rat)==3:   nrow, ncol= 1,3
        elif len(rat)==2: nrow, ncol= 1,2
        else : nrow, ncol= 2,2

        fig1,ax1 = plt.subplots(nrow,ncol,figsize=(10,10),dpi=100)
        ax1=ax1.ravel()





        # ============ Loop on ratios
        for i in range(0,len(rat)):

            # ============ Linear regression model

            S=stand.loc[stand[T[i]].isna()==False]  # Remove standards for which we don't have true measurements of the T[i] ratio
            S=S.loc[S[rat[i]+' Ratio'].isna()==False] # Remove standards for which we don't have NanoSIMS measurements of the T[i] ratio




            # ============ Iterative Plots ===========#


            ax1[i].set_title(T[i],fontsize=FT)

            # !!!! Probleme avec yerr=stand['ER-'+T[i]] !!! #
            ax1[i].errorbar(stand[rat[i]+' Ratio'],stand[T[i]],xerr=2*stand[rat[i]+' Ratio']*(stand[rat[i]+' Err Mean'])/100,
                      ecolor=C[i],elinewidth=2,marker='o',markersize=8, markerfacecolor=C[i],markeredgecolor='k',
                      linestyle='None',zorder=2)

            ax1[i].set_xlabel(rat[i])

            ax1[i].set_box_aspect(1)

            if i!=1:
                ax1[i].set_ylabel('True concentration (ppm)')
            else:
                ax1[i].set_ylabel('True concentration (wt%)')




        # ============ Out of loop

        #Modification of C/Si plot to change x axis format label into scientific notation
        ax1[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))



        # ============ Display plot within GUI interface
##        plt.show()  # To plot outside of the GUI

        # plt.tight_layout(h_pad=2,w_pad=1.5,pad=1)
        # plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig1, master=self.plot_window)
        canvas.draw()
        canvas.tkcanvas.pack(fill='both', expand=1)
##        toolbar = NavigationToolbar2Tk(canvas, self.plot_window)
##        toolbar.update()
##        canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
##
##



#==============================================================================================================#
#=========================================== New TreeView Function ============================================#
#==============================================================================================================#


    def New_Treeview(self,D):
        self.tv1["column"] = list(D.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["column"]:
            self.tv1.column(column, anchor='center')
            self.tv1.heading(column, text=column,anchor='center')

        data_rows=D.to_numpy().tolist()
        for row in data_rows:
            self.tv1.insert("","end", values=row)

        self.tv1.pack()
        return None


#==============================================================================================================#
#============================================= Clear Data Function ============================================#
#==============================================================================================================#

    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())
        for widget in self.plot_window.winfo_children():
            widget.destroy()
        return None


#==============================================================================================================#
#============================================== Save Files Function ===========================================#
#==============================================================================================================#

    def Save_Files(self):
        name=self.entry.get()
        data_corr.to_excel(os.path.dirname(filename)+'/'+name+'.xlsx', sheet_name='Python_corrected', index = False)
        fig1.savefig(os.path.dirname(filename)+'/'+name+'_std_linearregression.png', transparent=True)
        fig1.savefig(os.path.dirname(filename)+'/'+name+'_std_linearregression.pdf', transparent=True)
        return None


#==============================================================================================================#
#============================================= Switch Button Function =========================================#
#==============================================================================================================#

    def Switch_Event(self):
        if self.switch_var.get()=="off":
            X=stand_meas
            self.switch_1.configure(text="Standards")
        else:
            X=data
            self.switch_1.configure(text="Measurements")

        self.tv1.delete(*self.tv1.get_children())

        #Set up new treeview
        self.New_Treeview(X)
        return None


#==============================================================================================================#
#========================================== Regression Plot Function ==========================================#
#==============================================================================================================#


    def Reg_Plot(self):

        #Clear old treeview
        self.clear_data()

        self.linear_regression(stand_meas,data)

        #Set up new treeview
        self.New_Treeview(data)

        return None


#==============================================================================================================#
#==================================== Linear regression call function =========================================#
#==============================================================================================================#

    def linear_regression(self,stand_meas, data):

        global text_stat, data_corr, Er_comp, fig1,std_names


        #=======================================================================#
        #===================== Figure parameters definiton =====================#

        generalcolor='k'
        FT=12
        FT_L=10
        FC=generalcolor
        white=np.array([1,1,1,-0.25])

        plt.rcParams['axes.linewidth'] = 2 # Width of borders
        #plt.rcParams['axes.facecolor'] = 'k'
        plt.rcParams['axes.edgecolor'] = generalcolor
        plt.rcParams['lines.color'] = generalcolor
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 2
        plt.rcParams['xtick.labelsize'] = FT_L
        plt.rcParams['xtick.color'] = generalcolor
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['ytick.minor.width'] = 2
        plt.rcParams['ytick.labelsize'] = FT_L
        plt.rcParams['ytick.color'] = generalcolor


        #=======================================================================#
        #============== Confidence & Prediction Intervals function =============#

        def CIPI(X,newX,model,prediction,perc_interval):
            #X: data X axis
            #newX: Data on which Y has to be predicted
            #model: regression model on the data
            #prediction: Y value predicted on newX
            #perc_interval: wished percentage of confidence (usually 0.95 or 0.99)
            if perc_interval is None:
                tkinter.messagebox.showerror("No given percentage for confidence interval. Assuming 95% confidence interval.")
                perc_interval=0.95


            aveX=np.mean(X) #determines average value of X
            sx=np.std(X) #standard deviation of X
            T=t.ppf(perc_interval,len(X)-2) #Retrieving the t-value for a confidence interval of 95% with len(dat.sigmarat)-2 degrees of freedom
            s_size=model.scale**.5 # Residual standard error can also be calculated as np.sqrt(np.sum(model.resid**2)/model.df_resid)

            var=(1/len(X))+((newX-aveX)**2)/((len(X)-1)*sx**2)

            #Formula of Confidence interval can be found here: https://www.youtube.com/watch?feature=player_embedded&v=qVCQi0KPR0s
            CIpos=prediction+T*s_size*np.sqrt(var)
            CIneg=prediction-T*s_size*np.sqrt(var)
            #Prediction interval
            PIpos=prediction+T*s_size*np.sqrt(1+var)
            PIneg=prediction-T*s_size*np.sqrt(1+var)


            return CIneg, CIpos, PIneg, PIpos, T, s_size, aveX, len(X), sx

        #=======================================================================#
        #=========================== Data Extraction ===========================#

        #---------- Extracting standards true concentrations
        stand_true=pd.read_excel('Standards_concentrations.xlsx',header=0)


        #---------- Extracting data
        meas=data


        # stand_meas=pd.read_excel(name_datafile+'.xlsx',sheet_name='Std_Python',header=0)
        l=stand_meas.columns.to_list()
        stand=stand_meas[[m for m in l if 'Ratio' in m or 'Err Mean' in m]]
        stand.insert(0,"Name",stand_meas.Filename)


        del l

        std_names=[]

        ls_stand=stand_true.NAME.tolist()
        d=pd.DataFrame(columns=stand_true.columns.to_list())
        for i in range(0,len(stand)):
            X=stand.iloc[i]
            # l=[l for l in ls_stand if l in X.Name]
            Xname=re.sub(r'|'.join(('chain\d+_\d+','@\d+_\d+')),'',X.Name)
            l=difflib.get_close_matches(Xname,ls_stand,n=1,cutoff=0) # Retrieve good name of standards (even if variations are present)
            std_names.extend([l])
            d=pd.concat([d, stand_true.loc[stand_true.NAME==l[0]]], ignore_index=True)

        stand=pd.concat([stand,d],axis=1)

        del l, X, d


        #=======================================================================#
        #=================== General Least Square Regression ===================#

        rat=[re.sub(' Ratio','',i) for i in stand.columns if ' Ratio' in i]
        T=[re.sub(r'|'.join(('/28Si','\d+')),'',i) for i in rat]
        T=list(map(lambda x: x.replace('O', 'OH'), T))
        C=['tomato','royalblue','gold','mediumseagreen']

        if len(rat)==3:   nrow, ncol= 1,3
        elif len(rat)==2: nrow, ncol= 1,2
        else : nrow, ncol= 2,2

        fig1,ax1 = plt.subplots(nrow,ncol,figsize=(10,10),dpi=100)
        ax1=ax1.ravel()

        summary=np.zeros((4,7))
        text_stat=''


        if self.precinterval_var.get()==0:
            perc_int=0.95
            sig_value=2
        elif self.precinterval_var.get()==1:
            perc_int=0.99
            sig_value=3



        # ============ Loop on ratios
        for i in range(0,len(rat)):

            # ============ Linear regression model


            S=stand.loc[stand[T[i]].isna()==False]  # Remove standards for which we don't have true measurements of the T[i] ratio
            S=S.loc[S[rat[i]+' Ratio'].isna()==False] # Remove standards for which we don't have NanoSIMS measurements of the T[i] ratio
            X=np.linspace(0,np.max(S[rat[i]+' Ratio']),100)
            X_matrix=sm.add_constant(X)

            reg=sm.GLS(S[T[i]],sm.add_constant(S[rat[i]+' Ratio'])) # instanciation of General Least-Square.
            mod=reg.fit() # Launching linear regression
            predict=reg.predict(mod.params,X_matrix) #punctual prediction by applying the regression coefficients to estimate dilution on grain size
            CIneg,CIpos,PIneg,PIpos, T_test, resid_st_error, Average, N, std= CIPI(S[rat[i]+' Ratio'], X, mod, predict,perc_int) # Determine CI and PI limits



            # ============ Iterative Plots ===========#


            ax1[i].set_title(T[i],fontsize=FT)

            # !!!! Probleme avec yerr=stand['ER-'+T[i]] !!! #
            ax1[i].errorbar(stand[rat[i]+' Ratio'],stand[T[i]],xerr=2*stand[rat[i]+' Ratio']*(stand[rat[i]+' Err Mean'])/100,
                      ecolor=C[i],elinewidth=2,marker='o',markersize=8, markerfacecolor=C[i],markeredgecolor='k',
                      linestyle='None',zorder=2)


            ax1[i].plot(X,predict,'--k',markersize=12,zorder=1)
            ax1[i].plot(X,CIpos,'-.',color=C[i],label='_nolegend_')
            ax1[i].plot(X,CIneg,'-.',color=C[i],label='_nolegend_')
            ax1[i].plot(X,PIpos,'--',color=C[i],label='_nolegend_')
            ax1[i].plot(X,PIneg,'--',color=C[i],label='_nolegend_')
            ax1[i].fill_between(X, CIpos, CIneg,color=C[i],alpha=0.5,label='_nolegend_')

            ax1[i].text(ax1[i].get_xlim()[1]*0.05,ax1[i].get_ylim()[1]*0.90,'$R^{2}$= '+str(np.round(mod.rsquared,3)))
            ax1[i].text(ax1[i].get_xlim()[1]*0.05,ax1[i].get_ylim()[1]*0.85,r'$True = Meas \times$ '+str(int(np.round(mod.params[1])))+' + '+str(int(np.round(mod.params[0]))))

            ax1[i].set_xlabel(rat[i])

            ax1[i].set_box_aspect(1)

            if i!=1:
                ax1[i].set_ylabel('True concentration (ppm)')
            else:
                ax1[i].set_ylabel('True concentration (wt%)')



            # ============ Saving text strings containing statistical tests summaries
            text_stat=text_stat+'\n\n'+'----------------------------------------'+T[i]+'-----------------------------------'+'\n\n'+str(mod.summary())


            # ============ Saving parameters of linear regression
            summary[i,0]= mod.params[1] # slope
            summary[i,1]= mod.params[0] # origin
            summary[i,2]= T_test
            summary[i,3]= resid_st_error
            summary[i,4]= Average
            summary[i,5]= N
            summary[i,6]= std



        # ============ Out of loop

        #Modification of C/Si plot to change x axis format label into scientific notation
        ax1[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))



        # ============ Display plot within GUI interface
##        plt.show()  # To plot outside of the GUI

        plt.tight_layout(h_pad=2,w_pad=1.5,pad=1)
        # plt.tight_layout()
        canvas= FigureCanvasTkAgg(fig1, master=self.plot_window)
        canvas.draw()
        canvas._tkcanvas.pack(fill='both', expand=1)
##        toolbar = NavigationToolbar2Tk(canvas, self.plot_window)
##        toolbar.update()
##        canvas._tkcanvas.pack(fill=tkinter.BOTH, expand=1)
##
##
##        self.plot_window.pack(fill=tkinter.BOTH, expand=1)


        #=======================================================================#
        #==================== Volatils content calculations ====================#


        if self.buttonsCI_var.get()==0: #If Confidence Interval is selected for error propagation
            cste=0
        else: #If Prediction Interval is selected for error propagation
            cste=1

        volatils=np.zeros((len(meas),8))
        Er_comp=np.zeros((len(meas),20))
        Var=pd.DataFrame()
        sig=[]
        Er_lc=[]
        lc=[]

        for i in range(0,len(rat)):

            if T[i]=='OH': nround=4
            else: nround=1

            volatils[:,i*2]=np.round(meas[rat[i]+' Ratio']*summary[i,0]+summary[i,1],nround) # Calculation of volatils i content
            Er_comp[:,i*5]=np.round(meas[rat[i]+' Ratio']*summary[i,0]+summary[i,1],nround) # Calculation of volatils i content but saved in Er_comp

            #------------- Error calculation
            V=cste+1/summary[i,5]+(meas[rat[i]+' Ratio']-summary[i,4])**2/((summary[i,5]-1)*summary[i,6]**2) # Calcul variance to each measurement (not really variance)
            Var=pd.concat([Var,pd.DataFrame(V)],axis=1) # Store variance

            s=summary[i,2]*summary[i,3]*np.sqrt(V)/sig_value # Calculate error on prediction
            sig.append(s) # Store error on prediction

            Er_comp[:,1+i*5]=np.round(meas[rat[i]+' Err Mean'],1)
            Er_comp[:,2+i*5]=np.round(sig[i]/Er_comp[:,i*5]*100,2)
            Er_comp[:,3+i*5]=np.round(np.sqrt((meas[rat[i]+' Err Mean']/100)**2+(sig[i]/Er_comp[:,i*5])**2)*100,2)
            Er_comp[:,4+i*5]=np.round(Er_comp[:,i*5]*np.sqrt((meas[rat[i]+' Err Mean']/100)**2+(sig[i]/Er_comp[:,i*5])**2),nround)


            volatils[:,i*2+1]=volatils[:,i*2]*np.sqrt((meas[rat[i]+' Err Mean']/100)**2+(sig[i]/volatils[:,i*2])**2) #Absolute Error

            if T[i]=='C':
                L='CO2'
                unit='ppm'
            elif T[i]=='OH':
                L='H2O'
                unit='wt %'
            else:
                L=T[i]
                unit='ppm'

            Er_lc.extend([L+'('+unit+')','Er_ratio '+L+ '(%)', 'Er_regression '+L+' (%)', 'Er_corr '+L+ '(%)','Er_corr '+L+' ('+unit+')'])
            lc.extend([L+' ('+unit+')','sig'])

        Er_comp=pd.DataFrame(Er_comp,columns=Er_lc)
        Er_comp=pd.concat([meas.Filename,Er_comp],axis=1)

        volatils=pd.DataFrame(volatils,columns=lc)

        data_corr=pd.concat([meas,volatils,Er_comp],axis=1)







#==============================================================================================================#
#===================================== Stat Information Window Function =======================================#
#==============================================================================================================#

    def test_summaries(self):

        W=int(np.round(App.HEIGHT*0.4/0.75))
        H=int(np.round(App.HEIGHT*0.6/0.75))

        # If the box is unchecked by the user then the window is closed
        if self.check_var.get()=="off":
            self.stat_window.destroy()
        else: # If the box is checked by the user then
            if 'text_stat' not in globals():
                tkinter.messagebox.showerror("Information","You need to run the program on data first")
                self.check_box_testsummary.toggle()
            else:
                self.stat_window = customtkinter.CTkToplevel(self) #It creates a top level window
                self.stat_window.geometry(f"{W}x{H}") # with those dimensions (width x height)
                self.stat_window.title("Tests summaries") # and this name

                textbox=customtkinter.CTkTextbox(master=self.stat_window,fg_color="black",text_color="white")
                textbox.grid(row=0, column=0)
                textbox.insert("0.0",text_stat)
                text=textbox.get("0.0","end")
                textbox.configure(state="disabled")  # configure textbox to be read-only
                textbox.pack(side="top",fill="both",expand=True, padx=10,pady=10)

                # If the user closes the window using the X top right button then close the window and uncheck the button
                self.stat_window.protocol("WM_DELETE_WINDOW", lambda : [self.check_box_testsummary.deselect(), self.stat_window.destroy()])


        return None



#==============================================================================================================#
#========================================== Stat Logic Window Function ========================================#
#==============================================================================================================#

    def test_logic(self):

        W=int(np.round(App.WIDTH*0.3/0.6))
        H=int(np.round(App.HEIGHT*0.6/0.75))

        # If the box is unchecked by the user then the window is closed
        if self.check_var_testlogic.get()=="off":
            self.statlogic_window.destroy()
        else: # If the box is checked by the user then
            self.statlogic_window = customtkinter.CTkToplevel(self) #It creates a top level window
            self.statlogic_window.geometry(f"{W}x{H}") # with those dimensions (width x height)
            self.statlogic_window.title("Tests Logic") # and this name

            # In which the text summaries will be incorporated
            with open('test_logic.txt','r', encoding="utf8") as file:
                text_logic = file.read()

            textbox_Tlogic=customtkinter.CTkTextbox(master=self.statlogic_window,fg_color="black",text_color="white")
            textbox_Tlogic.grid(row=0, column=0)
            textbox_Tlogic.insert("0.0",text_logic)
            text_Tlogic=textbox_Tlogic.get("0.0","end")
            textbox_Tlogic.configure(state="disabled")  # configure textbox to be read-only
            textbox_Tlogic.pack(side="top",fill="both",expand=True, padx=10,pady=10)


            # If the user closes the window using the X top right button then close the window and uncheck the button
            self.statlogic_window.protocol("WM_DELETE_WINDOW", lambda : [self.check_box_testlogic.deselect(), self.statlogic_window.destroy()])


        return None


#==============================================================================================================#
#========================================= Error Recap Window Function ========================================#
#==============================================================================================================#

    def error_recap(self):

        global screen_width, screen_height

        W=int(np.round(App.WIDTH*0.3/0.6))
        H=int(np.round(App.HEIGHT*0.6/0.75))

        # If the box is unchecked by the user then the window is closed
        if self.check_var_error.get()=="off":
            self.error_window.destroy()
        else: # If the box is checked by the user then
            if 'Er_comp' not in globals():
                tkinter.messagebox.showerror("Information","You need to run the program on data first")
                self.check_box_error.toggle()
            else:
                self.error_window = customtkinter.CTkToplevel(self) #It creates a top level window
                self.error_window.geometry(f"{W}x{H}") # with those dimensions (width x height)
                self.error_window.title("Error Recap")

                self.tv_error=ttk.Treeview(master=self.error_window)
                self.tv_error.place(relheight=1,relwidth=1)
                treescrolly_error=customtkinter.CTkScrollbar(master=self.error_window, orientation="vertical", command=self.tv_error.yview)
                treescrollx_error=customtkinter.CTkScrollbar(master=self.error_window, orientation="horizontal", command=self.tv_error.xview)
                self.tv_error.configure(xscrollcommand=treescrollx_error.set, yscrollcommand=treescrolly_error.set)
                treescrollx_error.pack(side="bottom", fill="x")
                treescrolly_error.pack(side="right", fill="y")


                self.tv_error["column"] = list(Er_comp.columns)
                self.tv_error["show"] = "headings"
                for column in self.tv_error["column"]:
                    self.tv_error.column(column, anchor='center')
                    self.tv_error.heading(column,text=column,anchor='center')

                data_rows=Er_comp.to_numpy().tolist()
                for row in data_rows:
                    self.tv_error.insert("","end", values=row)

                self.tv_error.pack(fill="both",expand=True)


                # If the user closes the window using the X top right button then close the window and uncheck the button
                self.error_window.protocol("WM_DELETE_WINDOW", lambda : [self.check_box_error.deselect(), self.error_window.destroy()])


        return None





#==============================================================================================================#
#======================================== Already implemented functions =======================================#
#==============================================================================================================#

    def change_appearance_mode(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def on_closing(self, event=0):
        self.quit()

    def load_image(self, path, image_size):
        """ load rectangular image with path relative to PATH """
        return ImageTk.PhotoImage(Image.open(PATH + path).resize((image_size, image_size)))


if __name__ == "__main__":
    app = App()
    app.mainloop()
