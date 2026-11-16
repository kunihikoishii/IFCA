import datetime
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from phconvert import bhreader
except ImportError:
    messagebox.showerror("Import Error","This script requires the following external libraries: matplotlib, numpy, phconvert.")
    quit()

try:
    import IFCA
except ImportError:
    messagebox.showerror("Import Error","\"IFCA.py\" is not found, or its requirements (numba, scipy) are not fulfilled.")
    quit()

def filedialog_clicked():

    fTyp = [("B&H FIFO data file", ".spc")]
    iFile = filedialog.askopenfilename(title="Choose photon data", filetype = fTyp, initialdir = "./")
    path = os.path.dirname(iFile)
    filepath.set(path)
    file_name = os.path.basename(iFile)
    datafile.set(file_name)

def setupfiledialog_clicked():

    fTyp = [("B&H setup file", ".set")]
    iFile = filedialog.askopenfilename(title="Choose setup file", filetype = fTyp, initialdir = "./")
    setupfile.set(iFile)

def loadbutton_clicked():
    global d_tensor
    global num_files, photons, bin_num
    global d_eig

    path = filepath.get() + "/"
    file_name_pattern = datafile.get()
    if not file_name_pattern:
        messagebox.showerror("Error","Specify a photon data file.\n" + \
            "Wildcard characters (such as \"*\") may be used to choose multiple files simultaneously.")
        return
    file_list = list(Path(path).glob(file_name_pattern))
    if not file_list:
        messagebox.showerror("Error", "No matching file found.")
        return
    timestamps_unit = bhreader.load_spc(str(file_list[0]), "SPC-134")[3]
    param["clock"] = timestamps_unit
    param["gap"] = int(gap.get() / timestamps_unit)
    param["bin_time"] = bintime.get()
    param["tcspc_num_ch"] = numch.get()
    param["tcspc_bin_factor"] = binfactor.get()
    param["detector_list"] = [int(c) for c in detectorlistEntry.get().split(",")]

    loadButton.config(text="Now Loading...", state="disabled")
    loadButton.update()

    try:
        d_tensor, num_files, photons, bin_num = IFCA.accumulate_tensors(
            loader=bhreader.load_spc,
            args=("SPC-134",),
            path=path,
            file_name_pattern=file_name_pattern,
            **param
        )
    except Exception as e:
        print(e)
        loadButton.config(text="Load & Make Tensors", state="normal")
        loadButton.update()
        return

    loadButton.config(text="Load & Make Tensors", state="normal")
    loadButton.update()

    fig1 = plt.figure(figsize=(6,4))
    ax1_1 = fig1.add_subplot(1,1,1, title="Ensemble-averaged fluorescence pattern (EFP)")
    ax1_1.plot(range(1,len(d_tensor["vector"])+1),d_tensor["vector"]/param["bin_time"])
    ax1_1.set_xlabel("$x$")
    ax1_1.set_ylabel("Fluorescence intensity (counts s$^{-1}$)")
    ax1_1.margins(x=0)
    fig1.show()

    d_eig = IFCA.eigenvalue_analysis(d_tensor)

    fig2 = plt.figure(figsize=(12,4))
    ax2_1 = fig2.add_subplot(1,3,1, title="20 largest eigenvalues")
    ax2_1.plot(range(1,21), d_eig["eigenvalue"][:20], marker="o", linewidth=0)
    ax2_1.set_yscale("log")
    ax2_1.set_ylabel("Amplitude")
    ax2_1.set_xlabel("Eigenvalue number")
    ax2_1.set_xlim(left=0.1, right=20.9)
    ax2_2_0 = fig2.add_subplot(5,3,2, title="Eigenvectors (#1-5)")
    ax2_2_0.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,0])
    ax2_2_0.margins(x=0)
    ax2_2_0.set_xticks([])
    ax2_2_1 = fig2.add_subplot(5,3,5)
    ax2_2_1.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,1])
    ax2_2_1.margins(x=0)
    ax2_2_1.set_xticks([])
    ax2_2_2 = fig2.add_subplot(5,3,8)
    ax2_2_2.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,2])
    ax2_2_2.margins(x=0)
    ax2_2_2.set_xticks([])
    ax2_2_3 = fig2.add_subplot(5,3,11)
    ax2_2_3.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,3])
    ax2_2_3.margins(x=0)
    ax2_2_3.set_xticks([])
    ax2_2_4 = fig2.add_subplot(5,3,14)
    ax2_2_4.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,4])
    ax2_2_4.margins(x=0)
    ax2_2_4.set_xlabel("$x$")
    ax2_2_5 = fig2.add_subplot(5,3,3, title="Eigenvectors (#6-10)")
    ax2_2_5.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,5])
    ax2_2_5.margins(x=0)
    ax2_2_5.set_xticks([])
    ax2_2_6 = fig2.add_subplot(5,3,6)
    ax2_2_6.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,6])
    ax2_2_6.margins(x=0)
    ax2_2_6.set_xticks([])
    ax2_2_7 = fig2.add_subplot(5,3,9)
    ax2_2_7.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,7])
    ax2_2_7.margins(x=0)
    ax2_2_7.set_xticks([])
    ax2_2_8 = fig2.add_subplot(5,3,12)
    ax2_2_8.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,8])
    ax2_2_8.margins(x=0)
    ax2_2_8.set_xticks([])
    ax2_2_9 = fig2.add_subplot(5,3,15)
    ax2_2_9.plot(range(1,len(d_tensor["vector"])+1),d_eig["eigenvector"][:,9])
    ax2_2_9.margins(x=0)
    ax2_2_9.set_xlabel("$x$")
    fig2.show()

def findIFCbutton_clicked():
    global IFCA_result

    if not d_eig:
        messagebox.showerror("Error", "First, load data file(s) and make tensors.")
        return
    
    findIFCButton.config(text="Fitting...", state="disable")
    findIFCButton.update()

    try:
        IFCA_result = IFCA.find_IFC(**d_eig, n=numIFC.get(), maxiter=1000, tol=1e-13, trial=10)
    except Exception as e:
        print(e)
        findIFCButton.config(text="Find IFCs", state="normal")
        findIFCButton.update()
        return

    findIFCButton.config(text="Find IFCs", state="normal")
    findIFCButton.update()

    N_IFCA = IFCA_result["number_of_molecules_from_M&T"]
    D = IFCA_result["single_molecule_decay"]
    fig3 = plt.figure(figsize=(12, 4))
    ax3_1 = fig3.add_subplot(121, title="Concentrations of IFCs")
    ax3_1.grid(ls="--")
    ax3_1.bar(range(1,len(N_IFCA)+1), N_IFCA, align="center", width=0.5, label="Estimated from M&T")
    ax3_1.set_xticks(range(1,len(N_IFCA)+1))
    ax3_1.set_ylabel("$N$")
    ax3_1.set_xlabel("IFC number")
    ax3_1.set_axisbelow(True)
    ax3_1.grid(ls="--")
    ax3_2 = fig3.add_subplot(122, title="Single-molecule fluorescence patterns")
    ax3_2.plot(range(1,len(d_tensor["vector"])+1), D/param["bin_time"])
    ax3_2.legend(labels=["IFC #"+str(i) for i in range(1, D.shape[1]+1)])
    ax3_2.set_ylabel("Fluorescence intensity (counts s$^{-1}$ molecule$^{-1}$)")
    ax3_2.set_xlabel("$x$")
    ax3_2.set_axisbelow(True)
    ax3_2.grid(ls="--")
    ax3_2.margins(x=0)
    fig3.show()

def fittingresultbutton_clicked():

    if not IFCA_result:
        messagebox.showerror("Error", "First, find IFCs.")
        return
    
    n = numIFC.get()
    obs_1D = np.zeros(n*(n+1)*(n+2)//6)
    fit_1D = np.zeros(n*(n+1)*(n+2)//6)
    element = []
    index = 0
    for i in range(0, n):
        for j in range (0, i+1):
            for k in range(0, j+1):
                obs_1D[index] = IFCA_result["observed_tensor"][i, j, k]
                fit_1D[index] = IFCA_result["fitted_tensor"][i, j, k]
                element.append(str(tuple([i+1, j+1, k+1])))
                index += 1
    fig4 = plt.figure(figsize=(8,8))
    fig4.subplots_adjust(top=0.95, bottom=0.1, hspace=0.55)
    ax4_1 = fig4.add_subplot(3,1,1, title="Elements of reduced 3rd-order cumulant tensor")
    ax4_1.plot(element, obs_1D, "o", label='Data')
    ax4_1.plot(element, fit_1D, "x", label='Fit')
    ax4_1.set_ylabel("Amplitude")
    ax4_1.tick_params(axis="x", labelrotation=90)
    ax4_1.set_axisbelow(True)
    ax4_1.grid(ls="--")
    ax4_1.legend()
    ax4_1.margins(y=0.1)
    res_1D = np.zeros(n*(n+1)*(n+2)//6)
    res_1D = obs_1D - fit_1D
    ax4_2 = fig4.add_subplot(312, title="Fitting residuals")
    ax4_2.stem(element, res_1D)
    ax4_2.set_axisbelow(True)
    ax4_2.grid(ls="--")
    ax4_2.set_ylabel("Amplitude")
    ax4_2.set_ylim(-1.1*max(abs(res_1D)), 1.1*max(abs(res_1D)))
    ax4_2.tick_params(axis="x", labelrotation=90)
    TB = IFCA_result["observed_tensor"]
    inv_W = np.diag(1 / np.sqrt(d_eig["eigenvalue"][:n]))
    U = IFCA_result["rotation_matrix"]
    TBiWU = np.einsum("lmn,li,mj,nk->ijk", TB, inv_W @ U.T, inv_W @ U.T, inv_W @ U.T)
    TBiWU_1D = np.zeros(n*(n+1)*(n+2)//6)
    element = []
    index = 0
    for i in range(0, n):
        for j in range (0, i+1):
            for k in range(0, j+1):
                TBiWU_1D[index] = TBiWU[i, j, k]
                element.append(str(tuple([i+1, j+1, k+1])))
                index += 1
    ax4_3 = fig4.add_subplot(313, title="Elements of 'diagonalized' 3rd-order cumulant tensor")
    ax4_3.stem(element, TBiWU_1D)
    ax4_3.set_ylabel("Amplitude")
    ax4_3.tick_params(axis="x", labelrotation=90)
    ax4_3.set_axisbelow(True)
    ax4_3.grid(ls="--")
    ax4_3.margins(y=0.1)
    fig4.show()

def saveIFCbutton_clicked():

    if not IFCA_result:
        messagebox.showerror("Error", "First, find IFCs.")
        return
    
    fTyp = [("Comma-Separated Values", ".csv")]
    oFile = filedialog.asksaveasfilename(title="Save IFCs", filetypes = fTyp, initialdir = "./", defaultextension = "csv")
    if not oFile:
        return
    with open(oFile, "w") as f:
        np.savetxt(f, IFCA_result["single_molecule_decay"]/param["bin_time"], delimiter=",")

def saveNbutton_clicked():

    if not IFCA_result:
        messagebox.showerror("Error", "First, find IFCs.")
        return
    
    fTyp = [("Comma-Separated Values", ".csv")]
    oFile = filedialog.asksaveasfilename(title="Save N values", filetypes = fTyp, initialdir = "./", defaultextension = "csv")
    if not oFile:
        return
    with open(oFile, "w") as f:
        np.savetxt(f, IFCA_result["number_of_molecules_from_M&T"], delimiter=",")

def saveEFPbutton_clicked():

    if not IFCA_result:
        messagebox.showerror("Error", "First, find IFCs.")
        return
    
    fTyp = [("Comma-Separated Values", ".csv")]
    oFile = filedialog.asksaveasfilename(title="Save EFP", filetypes = fTyp, initialdir = "./", defaultextension = "csv")
    if not oFile:
        return
    with open(oFile, "w") as f:
        np.savetxt(f, d_tensor["vector"]/param["bin_time"], delimiter=",")

def execcmdbutton_clicked():

    iStr = simpledialog.askstring("Execute python command", "Command:".ljust(100))
    if iStr is not None:
        exec(iStr)

def reportbutton_clicked():

    if not IFCA_result:
        messagebox.showerror("Error", "Report can be created only after IFCA is completed.")
        return
    
    fig = plt.figure(figsize=(21/2.54, 29.7/2.54))

    text = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n"
    text += "Analyzed by: " + analyst.get() + "\n"
    fig.text(0.94, 0.95, text, ha="right", va="top", linespacing=1.5, size=10, wrap=True)
    text = "Data file: " + filepath.get() + "/" + datafile.get() + "\n"
    text += f"File(s) loaded: {num_files} (photons = {photons:,}; number of bins = {bin_num:,}; "
    text += f"total measurement time = {bin_num * param["bin_time"]:.2f} sec.)\n"
    try:
        setupinfo=bhreader.load_set(setupfile.get())
        text += "Data recorded: " + setupinfo["identification"]["Date"] + " " + setupinfo["identification"]["Time"]
        text += f" (TCSPC time/ch = {setupinfo["sys_params"]["SP_TAC_TC"]}"
        text += f" sec.; number of channels = {setupinfo["sys_params"]["SP_ADC_RE"]})\n"
    except Exception:
        pass
    text += f"IFCA settings: $T_\\mathrm{{bin}}$ = {param["bin_time"]} sec.; $T_\\mathrm{{c}}$ = {param["clock"]} sec.; $T_\\mathrm{{g}}$ = {gap.get()} sec.; "
    text += "detector list = [" + ",".join(map(str, param["detector_list"])) + "];\n"
    text += f"IFCA settings: TCSPC channel binning pattern = {param["tcspc_num_ch"]} / {param["tcspc_bin_factor"]}"
    try:
        text += f" (time/ch after binning = {setupinfo["sys_params"]["SP_TAC_TC"] * param["tcspc_bin_factor"]} sec.)\n"
    except Exception:
        text += "\n"
    fig.text(0.06, 0.91, text, ha="left", va="top", linespacing=1.5, size=10, wrap=True)

    fig.subplots_adjust(top=0.78, wspace=0.30, hspace=0.35)
    plt.rcParams["font.size"] = 8

    ax1 = fig.add_subplot(3,2,1, title="Ensemble-averaged fluorescence pattern")
    ax1.plot(range(1,len(d_tensor["vector"])+1),d_tensor["vector"]/param["bin_time"])
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("Fluorescence intensity (counts s$^{-1}$)")
    ax1.margins(x=0)

    ax2 = fig.add_subplot(3,2,2, title="Largest eigenvalues")
    ax2.plot(range(1,21), d_eig["eigenvalue"][:20], marker="o", linewidth=0)
    ax2.set_yscale("log")
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Eigenvalue number")
    ax2.set_xlim(left=0.1, right=20.9)

    ax3 = fig.add_subplot(3,2,3, title="Single-molecule fluorescence patterns")
    ax3.plot(range(1,len(d_tensor["vector"])+1), IFCA_result["single_molecule_decay"]/param["bin_time"])
    ax3.legend(labels=["IFC #"+str(i) for i in range(1, IFCA_result["single_molecule_decay"].shape[1]+1)])
    ax3.set_ylabel("Fluorescence intensity (counts s$^{-1}$ molecule$^{-1}$)")
    ax3.set_xlabel("$x$")
    ax3.set_axisbelow(True)
    ax3.grid(ls=":")
    ax3.margins(x=0)
    ax4 = fig.add_subplot(3,2,4, title="Concentrations of IFCs")
    ax4.grid(ls="--")
    ax4.bar(range(1,len(IFCA_result["number_of_molecules_from_M&T"])+1), IFCA_result["number_of_molecules_from_M&T"], align="center", width=0.5, label="Estimated from M&T")
    ax4.set_xticks(range(1,len(IFCA_result["number_of_molecules_from_M&T"])+1))
    ax4.set_ylabel("$N$")
    ax4.set_xlabel("IFC number")
    ax4.set_axisbelow(True)
    ax4.grid(ls=":")
    
    plt.rcParams["font.size"] = 10

    fTyp = [("Portable Document Format", ".pdf")]
    oFile = filedialog.asksaveasfilename(title="Publish report", filetypes = fTyp, initialdir = "./", defaultextension = "pdf")
    if not oFile:
        return
    fig.savefig(oFile)
    subprocess.Popen("start \"\" \""+oFile+"\"", shell=True)

if __name__ == "__main__":

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.it"] = "Arial:italic"
    plt.rcParams["mathtext.rm"] = "Arial"

    root = tk.Tk()
    root.title("IFCA for Becker&Hickl's .spc data files (SPC-130/140/150/160/830)")
    root.geometry("740x140")
    root.attributes("-topmost", True)

    frame0 = ttk.Frame(root, padding=5)
    frame0.grid(row=0, column=0, sticky=tk.W)

    bintimeLabel = ttk.Label(frame0, text="T_bin (sec.)", padding=(5, 2))
    bintimeLabel.pack(side = "left")

    bintime = tk.DoubleVar()
    bintime.set(5e-6)
    bintimeSpinbox = ttk.Spinbox(frame0, format="%g", from_=0, increment=1e-6, textvariable=bintime, to=1, width=8)
    bintimeSpinbox.pack(side = "left")

    gapLabel = ttk.Label(frame0, text="T_g (sec.)", padding=(5, 2))
    gapLabel.pack(side = "left")

    gap = tk.DoubleVar()
    gap.set(1.1e-7)
    gapSpinbox = ttk.Spinbox(frame0, format="%g", from_=0, increment=1e-6, textvariable=gap, to=1, width=8)
    gapSpinbox.pack(side = "left")

    detectorlistLabel = ttk.Label(frame0, text="Detector list", padding=(5, 2))
    detectorlistLabel.pack(side = "left")

    detectorlist = tk.StringVar()
    detectorlist.set("1, 0")
    detectorlistEntry = ttk.Entry(frame0, textvariable=detectorlist, width=10)
    detectorlistEntry.pack(side = "left")

    numchLabel = ttk.Label(frame0, text="TCSPC ch #", padding=(5, 2))
    numchLabel.pack(side = "left")

    numch = tk.IntVar()
    numch.set(4096)
    numchSpinbox = ttk.Spinbox(frame0, from_=1, increment=1, textvariable=numch, to=100000, width=6)
    numchSpinbox.pack(side = "left")

    binfactorLabel = ttk.Label(frame0, text="Bin factor", padding=(5, 2))
    binfactorLabel.pack(side = "left")

    binfactor = tk.IntVar()
    binfactor.set(32)
    binfactorSpinbox = ttk.Spinbox(frame0, from_=1, increment=1, textvariable=binfactor, to=1000, width=6)
    binfactorSpinbox.pack(side = "left")

    frame1 = ttk.Frame(root, padding=5)
    frame1.grid(row=1, column=0, sticky=tk.W)

    filepathLabel = ttk.Label(frame1, text="Path", padding=(5, 2))
    filepathLabel.pack(side = "left")

    filepath = tk.StringVar()
    filepathEntry = ttk.Entry(frame1, textvariable=filepath, width=40)
    filepathEntry.pack(side = "left")

    datafileLabel = ttk.Label(frame1, text="Photon data file", padding=(5, 2))
    datafileLabel.pack(side = "left")

    datafile = tk.StringVar()
    datafileEntry = ttk.Entry(frame1, textvariable=datafile, width=20)
    datafileEntry.pack(side = "left")

    filedialogButton = ttk.Button(frame1, text="Browse...", command=filedialog_clicked)
    filedialogButton.pack(side = "left")

    loadButton = ttk.Button(frame1, text="Load & Make Tensors", width=20, command=loadbutton_clicked)
    loadButton.pack(padx=10, side = "left")

    frame2 = ttk.Frame(root, padding=5)
    frame2.grid(row=2, column=0, sticky=tk.EW)

    numIFCLabel = ttk.Label(frame2, text="# of IFCs", padding=(2, 2))
    numIFCLabel.pack(side = "left")

    numIFC = tk.IntVar()
    numIFC.set(2)
    numIFCSpinbox = ttk.Spinbox(frame2, from_=2, increment=1, textvariable=numIFC, to=20, width=6)
    numIFCSpinbox.pack(side = "left")

    findIFCButton = ttk.Button(frame2, text="Find IFCs", width=10, command=findIFCbutton_clicked)
    findIFCButton.pack(padx=5, side = "left")

    seefittingresultButton = ttk.Button(frame2, text="See Fitting Results", command=fittingresultbutton_clicked)
    seefittingresultButton.pack(padx=5, side = "left")

    saveIFCButton = ttk.Button(frame2, text="Save IFCs", command=saveIFCbutton_clicked)
    saveIFCButton.pack(padx=5, side = "left")

    saveNButton = ttk.Button(frame2, text="Save N Values", command=saveNbutton_clicked)
    saveNButton.pack(padx=5, side = "left")

    saveEFPButton = ttk.Button(frame2, text="Save EFP", command=saveEFPbutton_clicked)
    saveEFPButton.pack(padx=5, side = "left")

    execcmdButton = ttk.Button(frame2, text="Execute Cmd", command=execcmdbutton_clicked)
    execcmdButton.pack(padx=10, side = "right")

    frame3 = ttk.Frame(root, padding=5)
    frame3.grid(row=3, column=0, sticky=tk.EW)

    analystLabel = ttk.Label(frame3, text="Analyst's name", padding=(5, 2))
    analystLabel.pack(side = "left")

    analyst = tk.StringVar()
    analystEntry = ttk.Entry(frame3, textvariable=analyst, width=20)
    analystEntry.pack(side = "left")

    setupfileLabel = ttk.Label(frame3, text="Setup file", padding=(5, 2))
    setupfileLabel.pack(side = "left")

    setupfile = tk.StringVar()
    setupfileEntry = ttk.Entry(frame3, textvariable=setupfile, width=20)
    setupfileEntry.pack(side = "left")

    setupfiledialogButton = ttk.Button(frame3, text="Browse...", command=setupfiledialog_clicked)
    setupfiledialogButton.pack(side = "left")

    reportButton = ttk.Button(frame3, text="Publish Report", command=reportbutton_clicked)
    reportButton.pack(padx=5, side = "left")

    quitButton = ttk.Button(frame3, text="Quit", command=quit)
    quitButton.pack(padx=10, side = "right")

    param = {}
    d_tensor = {}
    num_files = 0
    photons = 0
    bin_num = 0
    d_eig = {}
    IFCA_result = {}

    root.mainloop()
