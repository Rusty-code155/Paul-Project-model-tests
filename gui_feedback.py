# gui_feedback.py
# Code by Turner Miles Peeples
# gui_feedback.py
import tkinter as tk
from tkinter import ttk

def run_feedback_gui(model, group_id):
    root = tk.Tk()
    root.title("User Feedback")
    label = ttk.Label(root, text=f"Is the model prediction approaching your goal?\nGroup ID: {group_id}")
    label.pack(pady=10)
    ttk.Button(root, text="Yes", command=lambda: handle_response("yes")).pack(side=tk.LEFT, padx=20, pady=20)
    ttk.Button(root, text="No", command=lambda: handle_response("no")).pack(side=tk.RIGHT, padx=20, pady=20)
    
    def handle_response(response):
        model.update_score(response)
        root.destroy()

    root.mainloop()