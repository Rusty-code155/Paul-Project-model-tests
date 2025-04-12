# Code By Turner Miles Peeples
# main.py
from controller import run_training, run_testing
import tkinter as tk
from tkinter import ttk

def start_program_window():
    window = tk.Tk()
    window.title("Thermal Conductivity Predictor")
    window.geometry("300x200")
    ttk.Label(window, text="Choose Mode:", font=('Arial', 14)).pack(pady=10)
    ttk.Button(window, text="Train Models", command=lambda: [window.destroy(), run_training()]).pack(pady=10)
    ttk.Button(window, text="Test Models", command=lambda: [window.destroy(), run_testing()]).pack(pady=10)
    window.mainloop()

if __name__ == "__main__":
    start_program_window()