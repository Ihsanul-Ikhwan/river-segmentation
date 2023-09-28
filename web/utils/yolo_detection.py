import cv2
from utils.weight.checkfile import check_file


class Model:
    """Model class"""

    weight_name = "best.pt"
    # weight_name = "newbest.pt"
    # weight_name = "yolov8x-seg.pt"
    model = None
    ms = 0

    def __init__(self):
        self.model = check_file(self.weight_name)

    def detect(self, original_image):
        if self.model is None:
            raise FileNotFoundError(f"{self.weight_name} is not found!")

        try:
            # Run inference from input
            annotatedframe = self._generate_box(original_image)

            # Visualize results on the frame
            # annotatedframe = results[0].plot()

            # return as image
            if annotatedframe is None:
                annotatedframe = original_image

            _, jpg = cv2.imencode(".jpg", annotatedframe)
            return jpg.tobytes()
        except TypeError as e:
            print("Error: ", e)
            pass

    def _generate_box(self, image):
        try:
            results = self.model(image, device=0, show=False, conf=0.4)

            for result in results:
                box = result.boxes.xyxy.cpu().numpy().astype(int)
                if len(box) == 0:
                    continue
                if len(box[0]) == 0:
                    continue

                speed = (
                    result.speed["preprocess"]
                    + result.speed["inference"]
                    + result.speed["postprocess"]
                )

                LENGTH_ANCHOR = 338

                length = (-1 * box[0][1]) + LENGTH_ANCHOR

                if length <= 0:
                    self.ms = 0
                else:
                    self.ms += speed

                return self._draw(image, box[0], length, LENGTH_ANCHOR)
        except TypeError as e:
            print("errornya disini ", e)
            pass

    def _draw(self, frame, box, length, LENGTH_ANCHOR=332):
        spd_str = "{:.2f}".format(length / self.ms * 1000)
        base_est = ((LENGTH_ANCHOR - length) / (length / self.ms * 1000))  # in seconds

        minutes = base_est // 60
        seconds_left = base_est % 60

        est_str = ""

        est_str = (f"{minutes:2.0f}:{seconds_left:2.0f}", f"{base_est:2.0f}")[
            base_est < 60
        ]

        ms_str = "{:.0f}".format(self.ms // 1000)

        annotated_frame = cv2.line(
            frame, (347, LENGTH_ANCHOR), (347, 160), (255, 0, 0), 2
        )
        annotated_frame = cv2.line(
            annotated_frame, (347, LENGTH_ANCHOR), (347, box[1]), (0, 255, 0), 2
        )
        annotated_frame = cv2.rectangle(
            annotated_frame,
            (box[0], box[1]),
            (box[2], box[3]),
            (0, 0, 255),
            2,
        )

        annotated_frame = cv2.putText(
            annotated_frame,
            f"Panjang = {length} px",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        annotated_frame = cv2.putText(
            annotated_frame,
            f"elapsed {ms_str} s",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        annotated_frame = cv2.putText(
            annotated_frame,
            f"speed {spd_str} px/s",
            (50, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        annotated_frame = cv2.putText(
            annotated_frame,
            f"est {est_str}",
            (50, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return annotated_frame
