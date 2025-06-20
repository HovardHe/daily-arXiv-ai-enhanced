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

if os.path.exists('.env'):
    dotenv.load_dotenv()

try:
    template = open("template.txt", "r", encoding="utf-8").read()
    system = open("system.txt", "r", encoding="utf-8").read()
except FileNotFoundError as e:
    print(f"Error: Could not find template/system file. Make sure 'template.txt' and 'system.txt' are in the 'ai/' directory.", file=sys.stderr)
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input jsonline data file")
    return parser.parse_args()

def fix_json_escapes(json_str: str) -> str:
    result = []
    i = 0
    while i < len(json_str):
        char = json_str[i]
        if char == '\\':
            if i + 1 < len(json_str) and json_str[i+1] in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                result.append(char)
                result.append(json_str[i+1])
                i += 2
                continue
            result.append('\\\\')
            i += 1
        else:
            result.append(char)
            i += 1
    return "".join(result)

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')
    
    input_filepath = args.data
    output_filepath = input_filepath.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')

    if os.path.exists(output_filepath):
        os.remove(output_filepath)

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
    
    try:
        with open(input_filepath, "r", encoding="utf-8") as infile, \
             open(output_filepath, "a", encoding="utf-8") as outfile:
            
            lines = infile.readlines()
            total_items = len(lines)

            for idx, line in enumerate(lines):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item_id = item.get('id')
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                
                raw_response = ""
                
                try:
                    raw_response = chain.invoke({
                        "language": language,
                        "content": item.get('summary', '')
                    })

                    json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
                    if not json_match:
                        raise ValueError("No valid JSON object found in the LLM response.")
                    
                    json_str = json_match.group(0)

                    try:
                        ai_dict = json.loads(json_str)
                    except json.JSONDecodeError:
                        fixed_json_str = fix_json_escapes(json_str)
                        ai_dict = json.loads(fixed_json_str)
                    
                    item['AI'] = ai_dict

                except Exception as e:
                    print(f"An error occurred while processing ID {item_id}: {e}", file=sys.stderr)
                    print(f"Problematic raw response was: {raw_response}", file=sys.stderr)
                    item['AI'] = {
                         "tldr": "Error processing this paper.",
                         "motivation": "Error processing this paper.",
                         "method": "Error processing this paper.",
                         "result": "Error processing this paper.",
                         "conclusion": "Error processing this paper."
                    }
                
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"Finished item {idx + 1}/{total_items}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
