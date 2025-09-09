'''
v2在v1的基础上,扩展了可以解析任何格式的文件，包括pdf、ppt、doc、png的文件
'''
import requests
import base64
import json
import os
import cv2
from doclayout_yolo import YOLOv10
from pdf2image import convert_from_path
from pathlib import Path
from dotenv import load_dotenv
from utils import parse_2_json, parse_2_mds, get_resource, get_file_from_path
import subprocess
import shutil

class DocumentAnalyzer:
    def __init__(self, model_path: str):
        """初始化文档分析器
        Args:
            model_path: YOLO模型路径
        """
        self.model = YOLOv10(model_path)

    def pdf_to_images(self, pdf_path: str, output_dir: str,max_pages: int = 5):
        """将PDF转换为图像"""
        os.makedirs(output_dir, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=200)
        images_paths = []
        # 清理目录
        images = images[:max_pages] #默认只取前5页，如果要取全部，则设置为-1

        for f in os.listdir(output_dir):
            p = os.path.join(output_dir, f)
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            elif os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(output_path, "png")
            images_paths.append(output_path)
        return images_paths

    def _office_to_pdf(self, office_path: str, output_dir: str) -> str:
        """使用LibreOffice将office文件转换为PDF"""
        os.makedirs(output_dir, exist_ok=True)
        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            office_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"LibreOffice 转换失败: {proc.stderr.decode('utf-8', errors='ignore')}")
        pdf_name = Path(office_path).with_suffix(".pdf").name
        pdf_path = os.path.join(output_dir, pdf_name)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"未找到转换后的PDF: {pdf_path}")
        return pdf_path


    def ppt_to_images(self, ppt_path: str, output_dir: str, max_pages: int = 5):
        """将PPT,pptx转换为图像（先转PDF再转图）,ppt无法直接转png"""
        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, "_tmp_ppt_pdf")
        pdf_path = self._office_to_pdf(ppt_path, tmp_dir)
        try:
            return self.pdf_to_images(pdf_path, output_dir, max_pages)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def doc_to_images(self, doc_path: str, output_dir: str, max_pages: int = 5):
        """将DOC,docx转换为图像（先转PDF再转图）"""
        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, "_tmp_doc_pdf")
        pdf_path = self._office_to_pdf(doc_path, tmp_dir)
        try:
            return self.pdf_to_images(pdf_path, output_dir, max_pages)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def detect_layout(self, image_path, conf: float = 0.2):
        """使用YOLO检测文档布局"""
        image = cv2.imread(image_path)

        # yolo检测
        det_res = self.model.predict(
            image_path,
            imgsz=1024,
            conf=conf,
            device="cuda:0"
        )
        result = det_res[0]  # 一个 Results 对象
        return image, result

    def extract_questions_md(self,index,results, image, output_dir, vlm_client):
        '''
        提取所有题目为markdown格式
        先从上到下进行扫描，整页内容都保存为一个md文件，然后将md文件给到vlm进行处理
        '''
        #创建当前页的子文件夹
        page_dir = os.path.join(output_dir, f"page_{index}")
        os.makedirs(page_dir, exist_ok=True)
        #从上到下进行扫描，整页内容都保存为一个md文件
        # 取出boxes并排序
        items = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            cls_name = results.names[cls]
            conf = float(box.conf.cpu().numpy()[0])

            items.append({
                "class": cls_name,
                "bbox": (x1, y1, x2, y2),
                "conf": conf
            })

        # 按纵坐标排序，纵坐标相同的按横坐标排序
        items = sorted(items, key=lambda x: (x["bbox"][1],x["bbox"][0])) #先按y1排序，再按x1排序

        md_content = ""
        #遍历一遍items ,使用vlm模型提取所有内容，按照从上到下进行保存为markdown格式
        figure_queue = []
        for it in items:
            cls = it["class"]
            x1, y1, x2, y2 = it["bbox"]
            crop_img = image[y1:y2, x1:x2]
            crop_path = os.path.join(page_dir, f"{cls}_{x1}_{y1}_{x2}_{y2}.png")
            cv2.imwrite(crop_path, crop_img)

            #用vlm提取每个子图的内容，按顺序拼接，如果遇到figure，就保存其路径
            
            # 使用一个队列保存未配对的figure            
            import re
            if cls == "figure":
                # 立即追加figure图片到md_content
                # 计算图片相对于当前页面目录的相对路径，方便在markdown中引用图片
                rel_crop_path = os.path.relpath(crop_path, page_dir)
                # 先用默认名，后续caption再补充
                safe_caption = f"figure_{x1}_{y1}_{x2}_{y2}"
                md_content += f"![{safe_caption}]({rel_crop_path})\n"
                # 将figure信息和当前md_content长度放入队列，便于后续插入caption
                figure_queue.append({
                    "md_pos": len(md_content),  # 记录当前md_content长度
                    "safe_caption": safe_caption,
                    "rel_crop_path": rel_crop_path
                })
            elif cls == "abandon":
                continue
            elif cls == "figure_caption" and figure_queue:
                # 处理caption，插入到上一个figure后面
                extracted_text = vlm_client.process_image_str(crop_path)
                # 取出队首的figure信息
                figure_info = figure_queue.pop(0)
                # 清理caption内容，避免特殊字符
                safe_caption = re.sub(r'[\\/:*?"<>|]', '_', extracted_text.strip().replace('\n', ' '))
                if not safe_caption:
                    safe_caption = figure_info["safe_caption"]
                # 构造caption文本
                caption_text = f"{extracted_text}\n"
                # 在md_content中figure图片后插入caption
                insert_pos = figure_info["md_pos"]
                md_content = md_content[:insert_pos] + caption_text + md_content[insert_pos:]
            else:
                extracted_text = vlm_client.process_image_str(crop_path)  # 其他内容，按行拼接
                md_content += extracted_text
                md_content += "\n"


        # #将md内容保存为md文件
        md_path = os.path.join(page_dir, f"page_{index}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    def clean_questions_from_md(self,llm_client,questions_dir,questions_path):
        '''
        将md文件中的题目进行清洗，主要是题目和题图、以及对应的表格位置混乱，需要使用llm重新排版
        '''
        # 遍历questions_dir目录中的所有子目录（每个页面一个子目录）
        all_questions = []
        for index,page_dir in enumerate(os.listdir(questions_dir)):
            page_path = os.path.join(questions_dir, page_dir)
            if os.path.isdir(page_path):
                # 在每个页面目录中查找.md文件
                for file in os.listdir(page_path):
                    if file.endswith('.md'):
                        md_file_path = os.path.join(page_path, file)
                        print(f"正在处理文件: {md_file_path}")
                        md_content = llm_client.process_md_str(md_file_path,index)
                        #保存一整页的题目
                        with open(os.path.join(page_path, f"page_{index}.md"), "w", encoding="utf-8") as f:
                            f.write(md_content)
                        if md_content:
                            questions_list = parse_2_json(md_content)
                            # 打印时再格式化为字符串查看
                            print(f"清洗后的内容: {json.dumps(questions_list, ensure_ascii=False, indent=4)}")

                            # 累积所有题目
                            all_questions.extend(questions_list)
                            # 输出各题为md，并提供图片路径解析上下文
                            parse_2_mds(
                                questions_list,
                                questions_path,
                                base_dir=page_path,
                                search_root=questions_dir
                            )
        
        questions_json = os.path.join(questions_path, "all_questions.json")
        with open(questions_json, "w", encoding="utf-8") as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=4)


class QwenLLMClient:

    def __init__(self, api_key, api_url,model):
        """
        初始化Qwen-LLM客户端
        :param api_key: 百炼平台API密钥
        :param api_url: 百炼平台API接口地址
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }       


    def process_md_str(self, md_path,index):
        '''
        处理md文件内容,将含有多个题目的md文件给llm处理，主要是md文件中的内容题目和题图、以及对应的表格位置混乱，需要使用llm重新排版
        '''
        prompt = """
        你需要从一个 Markdown (.md) 文件中，**识别并提取所有完整的独立题目**。  
        文件中可能包含题干、题图（图片路径）、表格、公式等，且可能存在排版混乱（文字错位、多行分散、图文混排等）。  
        请严格按照以下规则处理：

        【规则】
        1. 完整性
        - 每道题必须包含完整题干、所有小问，以及相关的图片/表格/公式。
        - 小问（如 (1)、(2)、(a)、(b)）必须合并为一个整体题目。
        - **若题目不完整（题干缺失或残缺），直接忽略，不输出。**
        - **若题干中提到“某图/某表/某公式”，但文件中未找到对应的内容，则该题目直接忽略。**

        2. 独立性
        - 每道题作为独立单元输出，不得合并不同题目。
        - 图片、表格必须绑定到所属题目。

        3. 题图绑定
        - 若题图后紧跟 caption（如“习题2.16图”），根据 caption 绑定到对应题目编号（2.16）。
        - 若题图在题目前出现，但 caption 指向某题目，则绑定到该题目的 `images`。
        - 若题图在题目后出现，且 caption 指向某题目，则绑定到该题目。
        - 若题图无 caption，则视为位置正确，绑定到当前题目。

        4. 排除与过滤
        - 删除与题目无关的内容（标题、目录、页眉页脚、批注、无关说明等）。
        - 删除答案、解析、解题步骤、教学性讲解等非题目内容。
        - 删除重复出现的题目或片段。
        - **若文件中既有题目又有无关内容，必须保留题目并清理无关部分，而不是整体舍弃。**

        5. 编号规则
        - 使用纯数字带前缀的自增编号："{INDEX}_Q1", "{INDEX}_Q2", ...

        【输出格式】    
        输出为合法 JSON 数组，每个元素对应一道题目，格式如下：
        [
            {
                "id": "题目编号，纯数字带前缀的自增编号",
                "text": "题干完整文字",
                "images": ["图片路径1", "图片路径2", ...],
                "tables": ["表格内容1", "表格内容2", ...]
            }
        ]

        【注意】
        - 若题目没有图片或表格，则在 images/tables 中返回空数组。
        - 输出必须为合法 JSON，不得包含任何解释、注释或额外说明。
        - 若文件中确实没有符合要求的题目，则返回空数组 []。
        """
        prompt = prompt.replace("{INDEX}",str(index))
        
        with open(md_path, "r", encoding="utf-8") as f:
            md_str = f.read()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "text",
                            "text": md_str
                        }
                    ]
                }
            ],
            "temperature": 0.0,  # 降低随机性，确保排版准确性
            # "max_tokens": 1000,  # 增加最大输出长度，适应多题目排版
            # 可以添加的参数如下：
            # "top_p": 0.95,  # 控制采样多样性，通常与temperature配合使用
            # "presence_penalty": 0.0,  # 惩罚已出现过的token，提升新内容概率
            # "frequency_penalty": 0.0,  # 惩罚高频token，减少重复
            # "stop": ["\n\n"],  # 指定生成停止的标志，防止输出过多无关内容
            # "repetition_penalty": 1.0,  # 控制重复内容的惩罚力度
            # "logit_bias": {},  # 可对特定token的生成概率进行调整
            
        }
        
        #将md_str给到llm处理
        response = requests.post(
            self.api_url,
            headers=self.headers,
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                print("API返回结果格式不正确")
                print("完整响应:", result)
                return None
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print("错误信息:", response.text)
            return None

class QwenVLClient:
    def __init__(self, api_key, api_url,model):
        """
        初始化Qwen-VLM客户端
        :param api_key: 百炼平台API密钥
        :param api_url: 百炼平台API接口地址
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def encode_image(self, image_path):
        """
        将图片文件编码为带格式前缀的 base64 字符串（适配 VLM 模型）
        :param image_path: 图片文件路径（支持 JPG/JPEG/PNG/BMP）
        :return: 带格式前缀的 base64 字符串（如 data:image/png;base64,...），失败返回 None
        """
        # 1. 先校验图片格式（只支持常见图像格式，避免非图片文件）
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        file_ext = os.path.splitext(image_path)[1].lower()  # 获取文件后缀（小写）
        if file_ext not in valid_extensions:
            print(f"图片编码出错：不支持的格式「{file_ext}」，仅支持 {valid_extensions}")
            return None

        # 2. 校验文件是否存在
        if not os.path.exists(image_path):
            print(f"图片编码出错：文件不存在「{image_path}」")
            return None

        # 3. 校验文件大小（可选，避免过大文件导致 API 超限，可根据模型限制调整）
        max_file_size = 10 * 1024 * 1024  # 10MB（Qwen-VL 通常支持 10-30MB，可按需调整）
        file_size = os.path.getsize(image_path)
        if file_size > max_file_size:
            print(f"图片编码出错：文件过大（{file_size/1024/1024:.1f}MB），最大支持 {max_file_size/1024/1024:.1f}MB")
            return None

        # 4. 二进制读取并编码（核心逻辑）
        try:
            with open(image_path, "rb") as image_file:
                # 读取二进制数据
                image_bytes = image_file.read()
                # 生成 base64 字符串（不带前缀）
                base64_str = base64.b64encode(image_bytes).decode("utf-8")
                # 根据文件后缀添加格式前缀（关键：VLM 模型需要）
                if file_ext in (".jpg", ".jpeg"):
                    return f"data:image/jpeg;base64,{base64_str}"
                elif file_ext == ".png":
                    return f"data:image/png;base64,{base64_str}"
                elif file_ext == ".bmp":
                    return f"data:image/bmp;base64,{base64_str}"

        # 5. 细化异常捕获（便于定位问题）
        except PermissionError:
            print(f"图片编码出错：无权限读取文件「{image_path}」（请检查文件权限）")
        except IsADirectoryError:
            print(f"图片编码出错：「{image_path}」是目录，不是文件")
        except Exception as e:
            print(f"图片编码出错：未知错误 - {str(e)}（可能是文件损坏）")
        
        return None
    
    def process_image_str(self, image_path):
        """
        处理图片并识别题目内容
        :param image_path: 图片文件路径
        :return: 识别到的题目内容
        """
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            return None
        
        # 编码图片
        base64_image = self.encode_image(image_path)

        if not base64_image:
            return None
        
        prompt = """
                请精确识别图片中的所有题目内容，严格遵循以下要求：
                1. 完整提取所有信息，包括汉字、数字、字母、符号（如 +、-、×、÷、=、%、√ 等）和公式。
                2. 所有公式必须用 LaTeX 格式表示（例如：$E=mc^2$、$\\sum_{i=1}^n x_i$）。
                3. 保持题目原有的排版结构（如换行、分段、列表编号），不能合并或省略。
                4. 对于图片中的表格，必须转换为 **Markdown 表格格式**，并确保表格内容完整、无遗漏。
                5. 对于模糊或不确定的内容，用【】标注（例如：【可能是"3"或"5"】）。
                6. 必须提取图片中的所有内容（包括文字、表格、公式、符号等），不能遗漏。
                7. 输出仅包含识别结果的原文内容，不允许添加任何额外解释或说明。
                """
                
        # 构建请求数据
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,  # 温度参数，控制输出随机性
            # "max_tokens": 1000   # 最大输出 tokens
        }
        
        try:
            # 发送请求
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(payload)
            )
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                # 解析返回结果，根据实际API响应格式调整
                if "choices" in result and len(result["choices"]) > 0:
                    #返回一个字符串
                    return result["choices"][0]["message"]["content"]+"\n"
                else:
                    print("API返回结果格式不正确")
                    print("完整响应:", result)
                    return None
            else:
                print(f"API请求失败，状态码: {response.status_code}")
                print("错误信息:", response.text)
                return None
                
        except Exception as e:
            print(f"API调用出错: {str(e)}")
            return None


def open_debug_mode():
    import debugpy
    try:
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(("localhost", 9503))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass

if __name__ == "__main__":
    # open_debug_mode()     
    # url = "http://staff.ustc.edu.cn/~yjdeng/EM2022/hw/0414.pdf"
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    API_URL = os.getenv("API_URL")
    vlm_MODEL = os.getenv("vlm_MODEL")
    llm_MODEL = os.getenv("llm_MODEL")

    vlm_client = QwenVLClient(API_KEY, API_URL,vlm_MODEL)
    llm_client = QwenLLMClient(API_KEY, API_URL,llm_MODEL)
    #url
    pdf_url = ""
    ppt_url = ""
    doc_url = ""
    #path
    doc_path = ""


    yolo_model_path = "../yolo_model/doclayout_yolo_docstructbench_imgsz1024.pt"
    save_path = "../test_output/doc_output"
    questions_path = f"{save_path}/All_Questions"
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(questions_path, exist_ok=True)
    
    analyzer = DocumentAnalyzer(model_path=yolo_model_path)

    file_name, full_path = get_resource(doc_url, save_path)
    # file_name, full_path = get_file_from_path(doc_path)
    print(f"文件名: {file_name}, 保存路径: {full_path}")
    converted_images_dir = os.path.join(save_path, 'converted_images')

    questions_dir = os.path.join(save_path, 'questions')
    print("Converting file to images...")

    #判断文件是pdf、ppt、doc、png
    if file_name.endswith(".pdf"):
        image_paths = analyzer.pdf_to_images(full_path, converted_images_dir,2)
    elif file_name.endswith(".ppt") or file_name.endswith(".pptx"):
        image_paths = analyzer.ppt_to_images(full_path, converted_images_dir,5)
    elif file_name.endswith(".doc") or file_name.endswith(".docx"):
        image_paths = analyzer.doc_to_images(full_path, converted_images_dir)
    elif file_name.endswith(".png"):
        image_paths = analyzer.png_to_images(full_path, converted_images_dir)
    else:
        print("不支持的文件格式")
        exit()

    print(f"Converted {len(image_paths)} pages.")


    for i, image_path in enumerate(image_paths):
        image, result = analyzer.detect_layout(image_path, conf=0.2)
        # 保存检测结果
        annotated_image = result.plot(pil=True, line_width=3, font_size=20)
        cv2.imwrite(os.path.join(save_path, f"annotated_page_{i+1}.png"), annotated_image) 
        analyzer.extract_questions_md(i,result, image, questions_dir, vlm_client)
    
    # 所有页面处理完成后，统一清洗题目
    print("开始清洗题目...")
    analyzer.clean_questions_from_md(llm_client,questions_dir,questions_path)
