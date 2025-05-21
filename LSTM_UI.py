import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

''' 初始化主界面 '''
# 初始化主窗口
root = tk.Tk()
root.title("EEG Speech Classification")

# 设置主窗口布局（可以使用Grid或者Pack）
label_user = tk.Label(root, text="Current User: None")
label_user.pack()


''' 显示EEG图像的区域 '''
def show_eeg_image(eeg_data):
    fig, ax = plt.subplots()
    ax.plot(eeg_data)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()


''' 按钮bottom及其响应函数 '''
def on_register_click():
    username = entry_username.get()
    # 检查用户名是否存在
    if check_user_existence(username):
        label_user.config(text=f"Current User: {username}")
    else:
        messagebox.showinfo("Error", "Sorry, service not available.")

register_button = tk.Button(root, text="Register", command=on_register_click)
register_button.pack()
