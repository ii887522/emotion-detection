import tkinter as tk

def main():
    # How to resize all image datasets so that they are the same sizes ?

    window = tk.Tk()
    window.title("Emotion Detection")
    window.geometry("720x720")
    window.resizable(False, False)
    window.mainloop()

if __name__ == "__main__":
    main()
