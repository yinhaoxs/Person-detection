import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .models.common import DetectMultiBackend
from .utils.general import check_img_size, non_max_suppression
from .utils.augmentations import letterbox
import cv2
import base64
from io import BytesIO
import sys


class PersonDetect():
    def __init__(self, gpuid=0):
        self.device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() and gpuid != -1 else "cpu")

        root = os.path.dirname(__file__)
        sys.path.append(root + "/../persondetect/")
        # sys.path.append(root + "/../persondetect/models/")
        model_path = root + "/../models/yolov5/"
        # 参数配置
        self.weights = os.path.join(model_path + "yolov5x.pt")
        self.data = os.path.join(model_path + "coco128.yaml")
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.classes = [0]
        self.agnostic_nms = False

        # 加载模型
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.model.to(self.device)
        self.model.eval()

    def inference(self, b64_image):
        image = Image.open(BytesIO(base64.b64decode(b64_image)))
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img = letterbox(image, self.imgsz, stride=self.stride, auto=self.pt)[0]
        height, width, channel = img.shape
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # 模型推理部分，使用nms非极大抑制
        body_lists, confidences, bboxes = [], [], []
        with torch.set_grad_enabled(False):
            pred = self.model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            for i, det in enumerate(pred):  # per image
                if len(det):
                    for c in det[:, -1]:  # .unique()
                        body_lists.append(self.names[int(c)])
                        confidences.append(float(det[:, -2:-1].float()[0]))

                    for c in det[:, :4].int():
                        bbox_list = c.tolist()
                        bbox_list[0] = bbox_list[0] / width
                        bbox_list[2] = bbox_list[2] / width
                        bbox_list[1] = bbox_list[1] / height
                        bbox_list[3] = bbox_list[3] / height
                        bboxes.append(bbox_list)

        result = {'body_lists': body_lists, 'confidences': confidences, 'bboxes': bboxes}
        # print('PersonDetect inference result=' + str(result))
        return result


if __name__ == '__main__':
    input_dir = "/Users/parkliu/Documents/python/persondetect/temp/"
    output_dir = "/Users/parkliu/Documents/python/persondetect/output/"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = PersonDetect(device)

    # List all files in the input directory
    img_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        result = predictor.inference(img)

        if result is not None and 'bboxes' in result:
            bboxes = result['bboxes']
            confidences = result['confidences']

            if bboxes is not None:
                # Draw rectangles on the image
                img_with_rectangles = img.copy()
                draw = ImageDraw.Draw(img_with_rectangles)

                # Specify the font and size
                font = ImageFont.load_default()  # You can replace this with your preferred font
                font_size = 32  # Adjust the font size as needed

                for bbox, confidence in zip(bboxes, confidences):
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline="red", width=2)
                    # Annotate with confidence and control font size
                    confidence_str = f"Confidence: {confidence[0]:.2f}" if isinstance(confidence, list) else f"Confidence: {confidence:.2f}"
                    draw.text((x1 * width, y1 * height), confidence_str, fill="green", font=font)
                # Save the image with rectangles
                output_path = os.path.join(output_dir, f"result_{img_file}")
                img_with_rectangles.save(output_path)

                print(f"Result for {img_file} ({img_path}) saved with rectangles at {output_path}")

                # Optionally, display the image with rectangles
                # plt.imshow(img_with_rectangles)
                # plt.show()
