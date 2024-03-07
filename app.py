import numpy as np
import gradio as gr
from inference_sdk import InferenceHTTPClient

css_code = """
    body {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStg0vMxjadGIL_IAh-X1owMZ1Bf8TA2fDacw&usqp=CAU");
    }
  """

def retina(image):

  CLIENT = InferenceHTTPClient(
    api_url="http://detect.roboflow.com",
    api_key="F5zUvItYqVcvv1j2UrCt"
   ) 

  result = CLIENT.infer(image, model_id="ret-frkqi/1")
  return result['top']

app = gr.Interface(fn=retina, inputs='image', outputs='label',css=css_code)

app.launch(debug=True)
