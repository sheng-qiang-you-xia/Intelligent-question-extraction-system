import json
import os
from pathlib import Path


def parse_2_json(text: str):
    cleaned = text.strip()
    # 可选：先移除 ```json / ``` 包裹
    try:

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    except Exception:
        print("json格式不正确，没有```json，尝试第二种解析方式")

    # 先用严格 JSON
    try:
        return json.loads(cleaned)
    except Exception:
        print("json格式不正确，没有[]，尝试第三种解析方式")

    # 再用 JSON5 放宽解析
    try:
        import json5
        return json5.loads(cleaned)
    except Exception:
        print("json格式不正确，没有{}")
        return []

def parse_2_mds(cleaned_json, output_path, base_dir=None, search_root=None):
        '''
        将前面大模型输出的结构化题目json文件，转换为一个一个的题目md文件。
        - output_path: 所有题目的 md 输出目录（如 All_Questions）
        - base_dir: 本批题目的来源页面目录（如 questions/page_0），用于解析相对图片路径
        - search_root: 可选的全局检索根目录（如 questions），当仅给出文件名时在此目录递归查找
        '''
        try:
            # 兼容：既支持字符串也支持对象
            if isinstance(cleaned_json, str):
                questions = json.loads(cleaned_json)
            else:
                questions = cleaned_json

            if not isinstance(questions, list):
                print("警告: LLM返回的不是JSON数组格式")
                return
            os.makedirs(output_path, exist_ok=True)

            def resolve_image_path(img_value):
                # 已是绝对路径
                if os.path.isabs(img_value) and os.path.exists(img_value):
                    return img_value
                # 先尝试基于 base_dir 解析
                if base_dir:
                    p = os.path.join(base_dir, img_value)
                    if os.path.exists(p):
                        return p
                # 如果包含相对片段（page_*/filename）并能从 output_path 推断
                candidate_from_output = os.path.join(output_path, img_value)
                if os.path.exists(candidate_from_output):
                    return candidate_from_output
                # 在 search_root 递归查找同名文件
                if search_root and os.path.isdir(search_root):
                    name_only = os.path.basename(img_value)
                    for root, _, files in os.walk(search_root):
                        if name_only in files:
                            return os.path.join(root, name_only)
                # 找不到则返回原值（后续以原值写入，便于排查）
                return img_value

            for question in questions:
                if not isinstance(question, dict):
                    print(f"警告: 题目格式不正确: {question}")
                    continue
                    
                id = question.get("id", "unknown")
                text = question.get("text", "")
                images = question.get("images", [])
                tables = question.get("tables", [])
                
                md_content = f"## {id}\n{text}\n"
                for image in images:
                    resolved_abs = resolve_image_path(image)
                    # 生成相对于 output_path 的相对路径，便于 All_Questions/*.md 正确引用
                    rel_for_md = os.path.relpath(resolved_abs, output_path) if os.path.isabs(resolved_abs) or os.path.exists(resolved_abs) else image
                    md_content += f"![{image}]({rel_for_md})\n"
                for table in tables:
                    md_content += f"{table}\n"
                md_content += "\n"
                md_path = os.path.join(output_path, f"{id}.md")
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(md_content)
                print(f"题目{id}已保存到{id}.md")
                
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"清理后的内容: {cleaned_json}")


def get_resource(url, save_path):
        """通过url获取pdf文件"""
        import requests

        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"开始下载文件: {url}")

            response = requests.get(url)
    
            response.raise_for_status()
            file_name = Path(url).name 
            full_path = save_dir / file_name

            with open(full_path, 'wb') as f:
                f.write(response.content)
            print(f"成功下载文件并保存到: {save_path}")
            return file_name, str(full_path)

        except requests.RequestException as e:
            print(f"下载失败: {e}")
            return None, None
        except Exception as e:
            print(f"发生错误: {e}")
            return None, None

def get_file_from_path(file_path):
    '''
    从文件路径中获取文件名和路径
    '''
    file_name = Path(file_path).name
    return file_name, file_path