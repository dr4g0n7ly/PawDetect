from fastai.vision.all import *
import gradio as gr

learn = load_learner('model.pkl')

def classify_img(img):
    pred,idx,probs = learn.predict(img)
    return str(pred + ",   probability: " + str(round(torch.max(probs).item(), 4)) )

image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['test0.jpg', 'test1.jpg', 'test2.jpg']

intf = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
