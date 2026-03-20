import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64

# 使用 AiHubMix 的配置
api_key = "<AIHUBMIX_API_KEY>"  # 换成你在 AiHubMix 生成的密钥
base_url = "https://aihubmix.com/v1"

client = OpenAI(api_key=api_key, base_url=base_url)

# 可选：图片宽高比（部分接口不支持 4K，多为 1:1）
aspect_ratio = "1:1"  # 可选: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

prompt = "给图片中的角色添加笑容。"

# 输入图片路径
image_path = "tupian.png"
if not os.path.exists(image_path):
    print(f"请先准备输入图片: {image_path}")
    exit(1)

with open(image_path, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="gemini-3-pro-image-preview",  
    messages=[
        {"role": "system", "content": f"aspect_ratio={aspect_ratio}"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
    modalities=["text", "image"],
)

# 解析并保存返回的图片
try:
    parts = response.choices[0].message.multi_mod_content
    if parts:
        for part in parts:
            if "text" in part:
                print(part["text"])
            if "inline_data" in part and "data" in part["inline_data"]:
                image_data = base64.b64decode(part["inline_data"]["data"])
                image = Image.open(BytesIO(image_data))
                filename = f"generated_{aspect_ratio.replace(':', '-')}.png"
                image.save(filename)
                print(f"图片已保存: {filename}")
    else:
        print("未收到有效响应")
except Exception as e:
    print(f"错误: {e}")