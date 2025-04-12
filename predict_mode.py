#Code By Turner Miles Peeples
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import os
from data_preprocessing import preprocess_dataframe
from excel_handler import get_group_id
from neural_network import ANNGCModel
from logger import log_feedback
import numpy as np

def run_predict_gui():
    root = tk.Tk()
    root.title("Predict Mode")
    root.geometry("600x400")

    file_label = ttk.Label(root, text="No file loaded")
    file_label.pack(pady=5)

    group_label = ttk.Label(root, text="Group ID:")
    group_label.pack()
    group_entry = ttk.Entry(root)
    group_entry.pack(pady=5)

    result_label = ttk.Label(root, text="")
    result_label.pack(pady=5)

    def load_file():
        path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if path:
            file_label.config(text=f"Loaded: {os.path.basename(path)}")
            file_label.path = path

    def run_prediction():
        if not hasattr(file_label, 'path'):
            messagebox.showerror("No file", "Please load an Excel file first.")
            return

        try:
            df = pd.read_excel(file_label.path)
            group_id = group_entry.get().strip()

            if not group_id:
                group_id = str(get_group_id(df))
            group_id = eval(group_id) if isinstance(group_id, str) else group_id

            model = ANNGCModel(group_id)
            X, y, _, scaler_x, scaler_y = preprocess_dataframe(df)

            predictions = model.predict(X)
            model.plot_prediction_vs_actual(X, y, predictions, scaler_x, scaler_y)
            result_label.config(text="Prediction Complete. Provide Feedback:")

            # Save prediction log
            log_feedback(group_id, model.get_score(), model.history)

            def feedback_yes():
                model.update_score("yes")
                model.save_model()
                log_feedback(group_id, model.get_score(), model.history)
                messagebox.showinfo("Thanks", "Model rewarded.")

            def feedback_no():
                model.update_score("no")
                model.save_model()
                log_feedback(group_id, model.get_score(), model.history)
                messagebox.showinfo("Thanks", "Model penalized.")

            feedback_frame = tk.Frame(root)
            feedback_frame.pack(pady=10)

            ttk.Button(feedback_frame, text="Yes ✅", command=feedback_yes).pack(side=tk.LEFT, padx=5)
            ttk.Button(feedback_frame, text="No ❌", command=feedback_no).pack(side=tk.LEFT, padx=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")

    ttk.Button(root, text="Load Excel File", command=load_file).pack(pady=10)
    ttk.Button(root, text="Run Prediction", command=run_prediction).pack(pady=5)

    root.mainloop()
