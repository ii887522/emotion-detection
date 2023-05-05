# Author: Yong Chi Min

import base64
from io import BytesIO
from PIL import Image
import numpy as np


def b64encode_image(image_arr: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(image_arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
