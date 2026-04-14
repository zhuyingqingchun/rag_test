#!/usr/bin/env python3
"""
最小化RAG演示脚本
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def load_document(doc_path):
    """加载文档内容"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG Demo for Flight Control Document")
    parser.add_argument("--doc_path", type=str, default="data/flight_control_doc.md")
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()
    
    # 使用CPU以避免GPU问题
    device = "cpu"
    print(f"Using device: {device}")
    
    # 加载文档
    print(f"Loading document: {args.doc_path}")
    document = load_document(args.doc_path)
    print(f"Document length: {len(document)} characters")
    
    # 加载模型
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()
    
    # 测试查询
    test_queries = [
        "飞控系统由哪些部分组成？",
        "飞控系统的工作原理是什么？",
        "飞控系统有哪些飞行模式？",
        "飞控系统的关键技术有哪些？",
        "飞控系统在无人机中有哪些应用？"
    ]
    
    print("\n" + "="*80)
    print("Testing minimal RAG system with flight control document")
    print("="*80)
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*60}")
        
        # 构建提示
        # 只使用文档的前1000个字符以避免长度问题
        doc_snippet = document[:1000] + "..."
        prompt = f"Document: {doc_snippet}\n\nQuestion: {query}\nAnswer:"
        
        # 编码提示
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码回答
        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 提取回答部分
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        
        print("Response:")
        print(answer)
    
    # 交互式查询
    print("\n" + "="*80)
    print("Interactive mode. Type 'exit' to quit.")
    print("="*80)
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        # 构建提示
        doc_snippet = document[:1000] + "..."
        prompt = f"Document: {doc_snippet}\n\nQuestion: {query}\nAnswer:"
        
        # 编码提示
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码回答
        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 提取回答部分
        answer_start = response.find("Answer:") + len("Answer:")
        answer = response[answer_start:].strip()
        
        print("\nResponse:")
        print(answer)


if __name__ == "__main__":
    main()
