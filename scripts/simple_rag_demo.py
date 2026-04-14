#!/usr/bin/env python3
"""
简化版RAG演示脚本
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import os


def load_document(doc_path):
    """加载文档内容"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Simple RAG Demo for Flight Control Document")
    parser.add_argument("--doc_path", type=str, default="data/flight_control_doc.md")
    parser.add_argument("--encoder_model", type=str, default="distilroberta-base")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载文档
    print(f"Loading document: {args.doc_path}")
    document = load_document(args.doc_path)
    print(f"Document length: {len(document)} characters")
    
    # 加载编码器
    print("Loading encoder model...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    encoder_model = AutoModel.from_pretrained(args.encoder_model).to(device)
    encoder_model.eval()
    
    # 加载解码器
    print("Loading decoder model...")
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_model)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
    decoder_model = AutoModelForCausalLM.from_pretrained(args.decoder_model).to(device)
    decoder_model.eval()
    
    # 测试查询
    test_queries = [
        "飞控系统由哪些部分组成？",
        "飞控系统的工作原理是什么？",
        "飞控系统有哪些飞行模式？",
        "飞控系统的关键技术有哪些？",
        "飞控系统在无人机中有哪些应用？"
    ]
    
    print("\n" + "="*80)
    print("Testing simple RAG system with flight control document")
    print("="*80)
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*60}")
        
        # 构建提示
        prompt = f"Document: {document}\n\nQuestion: {query}\nAnswer:"
        
        # 编码提示
        inputs = decoder_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output = decoder_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码回答
        response = decoder_tokenizer.decode(
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
        prompt = f"Document: {document}\n\nQuestion: {query}\nAnswer:"
        
        # 编码提示
        inputs = decoder_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(device)
        
        # 生成回答
        with torch.no_grad():
            output = decoder_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 解码回答
        response = decoder_tokenizer.decode(
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
