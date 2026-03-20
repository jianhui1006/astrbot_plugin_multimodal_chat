import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

# 创建客户端
client = OpenAI(
    api_key="<AIHUBMIX_API_KEY>",
    base_url="https://aihubmix.com/v1",
)

# 可选参数（OpenAI接口不支持 4K 图片，默认为 1K)
aspect_ratio = "2:3"   # 支持: 1:1（默认）, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

prompt = (
    "Da Vinci style anatomical sketch of a dissected Monarch butterfly. "
    "Detailed drawings of the head, wings, and legs on textured parchment with notes in English."
)

response = client.chat.completions.create(
    model="gemini-3.1-flash-image-preview",
    messages=[
        {"role": "system", "content": f"aspect_ratio={aspect_ratio}"},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ],
    modalities=["text", "image"]
)

# 保存图片 & 输出文本
try:
    parts = response.choices[0].message.multi_mod_content
    if parts:
        for part in parts:
            if "text" in part:
                print(part["text"])
            if "inline_data" in part:
                image_data = base64.b64decode(part["inline_data"]["data"])
                image = Image.open(BytesIO(image_data))
                filename = f"butterfly_{aspect_ratio.replace(':','-')}.png"
                image.save(filename)
                print(f"Image saved: {filename}")
    else:
        print("No valid multimodal response received.")
except Exception as e:
    print(f"Error: {str(e)}")
