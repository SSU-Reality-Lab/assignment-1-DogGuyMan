import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import hybrid  # 'hybrid.py' 파일이 같은 폴더에 있어야 합니다.
import json
import numpy as np
import os
import argparse
import threading


def error(message):
    """오류 메시지 박스를 띄우는 함수"""
    messagebox.showerror("Error", message)


class BaseFrame(tk.Frame):
    """모든 프레임의 기반이 되는 클래스"""

    def __init__(self, parent, root, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = root

    def set_status(self, text):
        """상태 메시지를 설정하는 (지금은 비어있는) 함수"""
        print(f"Status: {text}")

    def ask_for_image(self, img_name=None):
        """파일 열기 대화상자를 통해 이미지를 불러오는 함수"""
        if img_name is None:
            img_name = filedialog.askopenfilename(
                parent=self,
                filetypes=[('Image Files', '*.png *.jpg *.jpeg *.bmp')]
            )
        if img_name and os.path.isfile(img_name):
            return img_name, cv2.imread(img_name)
        return None, None


class ImageWidget(tk.Label):
    """OpenCV 이미지를 표시하는 기본 위젯"""

    def __init__(self, parent):
        tk.Label.__init__(self, parent)
        self.image = None

    def draw_cv_image(self, cv_image):
        if cv_image is None:
            return

        self.image = cv_image
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_photo = ImageTk.PhotoImage(image=pil_img)

        self.config(image=tk_photo)
        self.image_for_tk = tk_photo

    def has_image(self):
        return self.image is not None

    def get_image(self):
        return self.image

    def write_to_file(self, f, grayscale):
        if self.image is not None:
            if grayscale:
                cv2.imwrite(f, cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
            else:
                cv2.imwrite(f, self.image)


class ClickableImageWidget(ImageWidget):
    """클릭 좌표를 저장하는 기능이 추가된 이미지 위젯"""

    def __init__(self, parent):
        super().__init__(parent)
        self.clicked_points = []
        self.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        self.push_click_image_coordinates(event.x, event.y)

    def draw_new_image(self, cv_image):
        self.clicked_points = []
        self.draw_cv_image(cv_image)

    def push_click_image_coordinates(self, x, y):
        self.clicked_points.append([x, y])
        print(f"Clicked at: ({x}, {y}), Total points: {len(self.clicked_points)}")

    def get_clicked_points_in_image_coordinates(self):
        return [(p[1], p[0]) for p in self.clicked_points]

    def pop_click(self):
        if self.clicked_points:
            return self.clicked_points.pop()
        return None


class ImageAlignmentFrame(BaseFrame):
    def __init__(self, parent, root, template_file=None):
        super().__init__(parent, root)

        tk.Button(self, text='Load First Image', command=self.load_first).grid(row=0, column=0, sticky="we")
        tk.Button(self, text='Load Second Image', command=self.load_second).grid(row=0, column=1, sticky="we")
        tk.Button(self, text='Undo', command=self.undo).grid(row=0, column=2, sticky="we")
        tk.Button(self, text='Redo', command=self.redo).grid(row=0, column=3, sticky="we")
        tk.Button(self, text='View Hybrid', command=self.process_compute).grid(row=0, column=4, sticky="we")
        tk.Button(self, text='Save Correspondances', command=self.save_corr).grid(row=1, column=0, sticky="we")
        tk.Button(self, text='Load Correspondances', command=self.load_corr).grid(row=1, column=1, sticky="we")

        self.left_image_widget = ClickableImageWidget(self)
        self.left_image_widget.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self.right_image_widget = ClickableImageWidget(self)
        self.right_image_widget.grid(row=2, column=3, columnspan=2, sticky="nsew")

        self.left_image_name = None
        self.right_image_name = None
        self.left_redo_queue = []
        self.right_redo_queue = []
        self.grid_rowconfigure(2, weight=1)
        self.image_receiver = None
        self.template_file = template_file

    def process_template(self):
        if self.template_file is not None:
            def load_template_and_compute():
                self.load_corr(self.template_file)
                self.process_compute()

            def load_template_local():
                self.after(0, load_template_and_compute)

            threading.Thread(target=load_template_local).start()

    def load_first(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.left_image_widget.draw_new_image(img)
            self.left_image_name = img_name

    def load_second(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.right_image_widget.draw_new_image(img)
            self.right_image_name = img_name

    def load_corr(self, filename=None):
        if filename is None:
            filename = filedialog.askopenfilename(parent=self, filetypes=[('JSON File', '*.json')])
        if filename and os.path.isfile(filename):
            with open(filename, 'r', encoding='utf-8') as infile:
                conf = json.load(infile)
                self.load_first(conf['first_image'])
                self.load_second(conf['second_image'])
                for c in conf['first_image_points']:
                    self.left_image_widget.push_click_image_coordinates(int(c[0]), int(c[1]))
                for c in conf['second_image_points']:
                    self.right_image_widget.push_click_image_coordinates(int(c[0]), int(c[1]))
                self.set_status('Loaded from template ' + filename)

    def save_corr(self):
        filename = filedialog.asksaveasfilename(parent=self, filetypes=[('JSON File', '*.json')])
        if filename:
            conf = {
                'first_image': self.left_image_name,
                'second_image': self.right_image_name,
                'first_image_points': self.left_image_widget.get_clicked_points_in_image_coordinates(),
                'second_image_points': self.right_image_widget.get_clicked_points_in_image_coordinates()
            }
            with open(filename, 'w') as outfile:
                json.dump(conf, outfile, indent=2)
                self.set_status('Saved to template ' + filename)

    def undo(self):
        action = self.left_image_widget.pop_click()
        if action: self.left_redo_queue.append(action)
        action = self.right_image_widget.pop_click()
        if action: self.right_redo_queue.append(action)

    def redo(self):
        if self.left_redo_queue:
            action = self.left_redo_queue.pop()
            self.left_image_widget.push_click_image_coordinates(action[0], action[1])
        if self.right_redo_queue:
            action = self.right_redo_queue.pop()
            self.right_image_widget.push_click_image_coordinates(action[0], action[1])

    def get_mapping(self):
        if not (self.left_image_widget.has_image() and self.right_image_widget.has_image()):
            return None
        left = self.left_image_widget.get_clicked_points_in_image_coordinates()
        right = self.right_image_widget.get_clicked_points_in_image_coordinates()

        num_points = min(len(left), len(right))
        if num_points != 3:
            error('Please click on at exactly three corresponding points.')
            return None

        left = np.array([[x, y] for y, x in left[:num_points]], np.float32)
        right = np.array([[x, y] for y, x in right[:num_points]], np.float32)
        return cv2.getAffineTransform(right, left)

    def set_receiver(self, receiver):
        self.image_receiver = receiver

    def process_compute(self):
        mapping = self.get_mapping()
        if mapping is not None and self.image_receiver is not None:
            self.image_receiver(self.left_image_widget.get_image(), self.right_image_widget.get_image(), mapping)


class HybridImageFrame(BaseFrame):
    def __init__(self, parent, root, receiver, tab_num, config_file=None):
        super().__init__(parent, root)

        # --- 1. 전체 UI 컨트롤 복원 및 자동 업데이트 제거 ---

        # Left Image Controls
        tk.Label(self, text='Left Image Sigma:').grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.left_sigma_slider = tk.Scale(self, from_=0.1, to=20, resolution=0.1, orient='horizontal')
        self.left_sigma_slider.set(7.0)
        self.left_sigma_slider.grid(row=0, column=1, sticky='we')

        tk.Label(self, text='Left Image Kernel Size:').grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.left_size_slider = tk.Scale(self, from_=3, to=41, resolution=2, orient='horizontal')
        self.left_size_slider.set(15)
        self.left_size_slider.grid(row=1, column=1, sticky='we')

        self.left_high_low_indicator = tk.StringVar(value='low')
        tk.Radiobutton(self, text='High Pass', variable=self.left_high_low_indicator, value='high').grid(row=2,
                                                                                                         column=0,
                                                                                                         sticky='w',
                                                                                                         padx=20)
        tk.Radiobutton(self, text='Low Pass', variable=self.left_high_low_indicator, value='low').grid(row=2, column=1,
                                                                                                       sticky='w')

        # Right Image Controls
        tk.Label(self, text='Right Image Sigma:').grid(row=0, column=2, sticky='e', padx=5, pady=2)
        self.right_sigma_slider = tk.Scale(self, from_=0.1, to=20, resolution=0.1, orient='horizontal')
        self.right_sigma_slider.set(4.5)
        self.right_sigma_slider.grid(row=0, column=3, sticky='we', padx=(0, 10))

        tk.Label(self, text='Right Image Kernel Size:').grid(row=1, column=2, sticky='e', padx=5, pady=2)
        self.right_size_slider = tk.Scale(self, from_=3, to=41, resolution=2, orient='horizontal')
        self.right_size_slider.set(9)
        self.right_size_slider.grid(row=1, column=3, sticky='we', padx=(0, 10))

        self.right_high_low_indicator = tk.StringVar(value='high')
        tk.Radiobutton(self, text='High Pass', variable=self.right_high_low_indicator, value='high').grid(row=2,
                                                                                                          column=2,
                                                                                                          sticky='w',
                                                                                                          padx=20)
        tk.Radiobutton(self, text='Low Pass', variable=self.right_high_low_indicator, value='low').grid(row=2, column=3,
                                                                                                        sticky='w')

        # Mix-in and Scale Controls
        tk.Label(self, text='Mix-in Ratio (0=left, 1=right):').grid(row=3, column=0, columnspan=2, sticky='e', padx=5,
                                                                    pady=2)
        self.mixin_slider = tk.Scale(self, from_=0.0, to=1.0, resolution=0.05, orient='horizontal')
        self.mixin_slider.set(0.5)
        self.mixin_slider.grid(row=3, column=2, columnspan=2, sticky='we', padx=(0, 10))

        tk.Label(self, text='Scale Factor:').grid(row=4, column=0, columnspan=2, sticky='e', padx=5, pady=2)
        self.scale_slider = tk.Scale(self, from_=1.0, to=5.0, resolution=0.2, orient='horizontal')
        self.scale_slider.set(2.0)
        self.scale_slider.grid(row=4, column=2, columnspan=2, sticky='we', padx=(0, 10))

        # [20192460 홍상준] Sharpening Alpha Controls
        tk.Label(self, text='Sharpening Weight:').grid(row=5, column=0, columnspan=2, sticky='e', padx=5, pady=2)
        self.weight_slider = tk.Scale(self, from_=0, to=1.0, resolution=0.05, orient='horizontal')
        self.weight_slider.set(0.5)
        self.weight_slider.grid(row=5, column=2, columnspan=2, sticky='we', padx=(0, 10))

        # --- 2. 'Apply' 버튼 추가 ---
        self.apply_button = tk.Button(self, text="Apply Changes", command=self.update_hybrid)
        self.apply_button.grid(row=6, column=0, columnspan=4, sticky='we', padx=10, pady=10)

        # Image Display Widget
        self.image_widget = ImageWidget(self)
        self.image_widget.grid(row=7, column=0, columnspan=4, sticky="nsew")

        # Grid configuration
        self.grid_rowconfigure(6, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(3, weight=1)

        # Instance variables
        self.left_image = None
        self.right_image = None
        self.tab_num = tab_num
        self.parent = parent  # ttk.Notebook parent

        receiver.set_receiver(self.set_images_and_mapping)

    def set_images_and_mapping(self, img1, img2, mapping):
        self.left_image = img1
        h, w = img1.shape[:2]
        self.right_image = cv2.warpAffine(img2, mapping, (w, h), borderMode=cv2.BORDER_REFLECT)
        if self.tab_num >= 0:
            self.parent.tab(self.tab_num, state="normal")
            self.parent.select(self.tab_num)
        self.update_hybrid()  # 최초 한 번은 바로 적용

    def update_hybrid(self, *args):
        if self.left_image is not None and self.right_image is not None:
            # --- 3. UI 컨트롤의 실제 값을 읽어오도록 수정 ---
            sigma1 = self.left_sigma_slider.get()
            size1 = int(self.left_size_slider.get())
            mode1 = self.left_high_low_indicator.get()

            sigma2 = self.right_sigma_slider.get()
            size2 = int(self.right_size_slider.get())
            mode2 = self.right_high_low_indicator.get()

            mixin_ratio = self.mixin_slider.get()
            scale_factor = self.scale_slider.get()
            sharpness_weight = self.weight_slider.get()

            # 커널 사이즈는 항상 홀수여야 함
            if size1 % 2 == 0: size1 += 1
            if size2 % 2 == 0: size2 += 1

            print(f"Updating with: s1={sigma1}, k1={size1}, m1='{mode1}' | s2={sigma2}, k2={size2}, m2='{mode2}'")

            hybrid_image = hybrid.create_hybrid_image(
                self.left_image, self.right_image,
                sigma1, size1, mode1,
                sigma2, size2, mode2,
                mixin_ratio, scale_factor, sharpness_weight)
            self.image_widget.draw_cv_image(hybrid_image)


class HybridImagesUIFrame(tk.Frame):
    def __init__(self, parent, root, template_file=None, config_file=None):
        super().__init__(parent)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, sticky="nsew")

        alignment_frame = ImageAlignmentFrame(notebook, root, template_file)
        notebook.add(alignment_frame, text='Align Images')

        hybrid_frame = HybridImageFrame(notebook, root, alignment_frame, 1, config_file)
        notebook.add(hybrid_frame, text='View Hybrid')
        notebook.tab(1, state="disabled")

        alignment_frame.process_template()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run the Hybrid Images GUI.')
    parser.add_argument('--template', '-t', help='A template file.', default=None)
    parser.add_argument('--config', '-c', help='Configuration for generating the hybrid image.', default=None)
    args = parser.parse_args()

    root = tk.Tk()
    root.title('Hybrid Images Project')

    # Adjust window size to be reasonable
    w = int(root.winfo_screenwidth() * 0.8)
    h = int(root.winfo_screenheight() * 0.8)
    root.geometry(f'{w}x{h}+50+50')

    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    app = HybridImagesUIFrame(root, root, args.template, args.config)
    app.grid(row=0, sticky="nsew")

    root.mainloop()