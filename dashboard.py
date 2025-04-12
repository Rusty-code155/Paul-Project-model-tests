#Code By Turner Miles Peeples
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from matplotlib.backends.backend_pdf import PdfPages
import ast

def plot_score_over_time(df, group_id):
    group_data = df[df["Group ID"] == group_id]
    timestamps = pd.to_datetime(group_data["Timestamp"])
    scores = group_data["Score"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(timestamps, scores, marker='o')
    ax.set_title(f"Score Over Time – Group {group_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Score")
    ax.grid(True)
    fig.autofmt_xdate()
    return fig

def plot_loss_history(df, group_id):
    group_data = df[df["Group ID"] == group_id]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, row in group_data.iterrows():
        try:
            loss_list = ast.literal_eval(row["Loss History"])
            ax.plot(loss_list, label=str(row["Timestamp"])[:19])
        except:
            continue

    ax.set_title(f"Loss History – Group {group_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize='x-small')
    ax.grid(True)
    return fig

def export_summary_to_pdf(df, selected_ids):
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
    if not file_path:
        return

    with PdfPages(file_path) as pdf:
        for group_id in selected_ids:
            fig1 = plot_score_over_time(df, group_id)
            fig2 = plot_loss_history(df, group_id)
            pdf.savefig(fig1)
            pdf.savefig(fig2)
            plt.close(fig1)
            plt.close(fig2)

    messagebox.showinfo("Export Complete", f"Summary exported to:\n{file_path}")

def run_dashboard():
    log_path = "logs/model_feedback_log.csv"
    if not os.path.exists(log_path):
        messagebox.showerror("Missing Log", "No feedback log found.")
        return

    df = pd.read_csv(log_path)

    root = tk.Tk()
    root.title("Model Dashboard")
    root.geometry("1000x700")

    ttk.Label(root, text="Select Group(s):").pack(pady=5)
    group_ids = sorted(df["Group ID"].unique())
    group_var = tk.StringVar(value=group_ids)

    group_listbox = tk.Listbox(root, listvariable=group_var, selectmode=tk.MULTIPLE, height=6)
    for g in group_ids:
        group_listbox.insert(tk.END, g)
    group_listbox.pack()

    plot_frame = tk.Frame(root)
    plot_frame.pack(pady=10, expand=True)

    def update_plots():
        for widget in plot_frame.winfo_children():
            widget.destroy()

        selection = [group_listbox.get(i) for i in group_listbox.curselection()]
        if not selection:
            messagebox.showinfo("Select Group", "Please select at least one group.")
            return

        for group in selection:
            fig1 = plot_score_over_time(df, group)
            fig2 = plot_loss_history(df, group)

            for fig in [fig1, fig2]:
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(pady=10)

    def export_selected():
        selected = [group_listbox.get(i) for i in group_listbox.curselection()]
        if not selected:
            messagebox.showinfo("Select Group", "Please select group(s) to export.")
            return
        export_summary_to_pdf(df, selected)

    ttk.Button(root, text="Update Plots", command=update_plots).pack(pady=5)
    ttk.Button(root, text="Export Summary to PDF", command=export_selected).pack(pady=5)

    update_plots()
    root.mainloop()
