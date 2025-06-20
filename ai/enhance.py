import os
import json
import sys
import re
import dotenv
import argparse

from langchain_openai import ChatOpenAI
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

# 当 .env 文件存在时，加载环境变量
if os.path.exists('.env'):
    dotenv.load_dotenv()

# 从外部文件读取指令模板
try:
    template = open("template.txt", "r", encoding="utf-8").read()
    system = open("system.txt", "r", encoding="utf-8").read()
except FileNotFoundError as e:
    print(f"Error: Could not find template/system file. Make sure 'template.txt' and 'system.txt' are in the 'ai/' directory.", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input jsonline data file")
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')
    
    input_filepath = args.data
    output_filepath = input_filepath.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')

    # --- 核心改动：流式处理，避免内存溢出 ---

    # 1. 每次运行时，先删除旧的输出文件，防止追加到不完整的文件上
    if os.path.exists(output_filepath):
        os.remove(output_filepath)

    # 2. 配置 LLM 链，直接输出字符串，不再尝试自动解析结构
    llm = ChatOpenAI(model=model_name)
    output_parser = StrOutputParser()
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])
    chain = prompt_template | llm | output_parser
    
    print('Open:', input_filepath, file=sys.stderr)
    print('Connect to:', model_name, file=sys.stderr)

    seen_ids = set()
    item_count = 0
    
    # 3. 同时打开输入和输出文件，逐行处理
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile, \
             open(output_filepath, "a", encoding="utf-8") as outfile:
            
            for line in infile:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed line in input file: {line.strip()}", file=sys.stderr)
                    continue

                # 去重逻辑
                item_id = item.get('id')
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                
                item_count += 1
                raw_response = ""
                
                try:
                    # 调用模型获取原始文本回复
                    raw_response = chain.invoke({
                        "language": language,
                        "content": item.get('summary', '')
                    })

                    # 从回复中贪婪地提取出最外层的 JSON 部分
                    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                    if not json_match:
                        raise ValueError("No valid JSON object found in the LLM response.")
                    
                    json_str = json_match.group(0)

                    # 尝试解析，如果失败则尝试修复
                    try:
                        ai_dict = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # 如果是无效的转义符错误，进行修复
                        if "Invalid \\escape" in str(e):
                            # 将所有未被正确转义的`\`替换为`\\`
                            fixed_json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                            ai_dict = json.loads(fixed_json_str)
                        else:
                            # 如果是其他JSON错误，直接抛出
                            raise e
                    
                    item['AI'] = ai_dict

                except Exception as e:
                    print(f"An error occurred while processing ID {item_id}: {e}", file=sys.stderr)
                    print(f"Problematic raw response was: {raw_response}", file=sys.stderr)
                    item['AI'] = {
                         "tldr": "Error", "motivation": "Error", "method": "Error", "result": "Error", "conclusion": "Error"
                    }
                
                # 4. 将处理好的单条记录立刻写入输出文件，并确保中文正常显示
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Finished item {item_count}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
