# import the inference-sdk
from inference_sdk import InferenceHTTPClient
import cv2
# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9ZEU5nKrT7cNBppGQlqY"
)

import supervision as sv

# infer on a local image
image_path = "Dataset_BUSI_with_GT/benign/benign (409).png"
result = CLIENT.infer(image_path, model_id="ultrasound-mxskc-l9mbr/1")

# Carrega a imagem original
image = cv2.imread(image_path)

# Converte o resultado da API para o formato do Supervision
detections = sv.Detections.from_inference(result)

# Cria os anotadores (para desenhar a máscara de segmentação e o rótulo)
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

# Aplica as anotações na imagem
annotated_image = mask_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Mostra a imagem com as predições
cv2.imshow("result", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
