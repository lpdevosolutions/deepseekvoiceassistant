from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import pipeline
import re
import torchvision
torchvision.disable_beta_transforms_warning()
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

CORS(app)

pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

@app.route('/api', methods=['POST'])
def api():
    print("##################################")
    user_message = request.json.get('message', '')
    print("#User Message:", user_message)
    print("##################################")
    
    if user_message == '':
        return jsonify({"response": "No message received."})

    messages = [{"role": "user", "content": user_message}]
    
    out = pipe(messages)

   
    generated_text = out[0].get('generated_text', [])
    

    if generated_text:
        assistant_response = generated_text[1]['content'].strip()
    else:
        assistant_response = "No response generated."

    cleaned_response = re.sub(r'<think>\n\n</think>\n\n', '', assistant_response)
    
    return jsonify({"response": cleaned_response})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
