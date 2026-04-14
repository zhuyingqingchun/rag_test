#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本切分模块
将文档切成 chunk，支持按段落或固定长度切分
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def split_by_paragraph(text: str, min_length: int = 10) -> List[str]:
    """按段落切分文本"""
    paragraphs = text.split('\n\n')
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if len(para) >= min_length:
            chunks.append(para)
    
    return chunks


def split_by_char(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """按字符数切分文本（支持重叠）"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def split_by_sentence(text: str, chunk_size: int = 512) -> List[str]:
    """按句子切分文本（尽量保持句子完整）"""
    import re
    
    sentences = re.split(r'[。！？!?]', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + '。'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '。'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_document(doc: Dict[str, Any], method: str = 'paragraph', 
                   chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """切分单个文档"""
    text = doc['text']
    doc_id = doc['doc_id']
    source = doc['source']
    
    if method == 'paragraph':
        chunks_text = split_by_paragraph(text)
    elif method == 'char':
        chunks_text = split_by_char(text, chunk_size, overlap)
    elif method == 'sentence':
        chunks_text = split_by_sentence(text, chunk_size)
    else:
        raise ValueError(f"不支持的切分方法: {method}")
    
    chunks = []
    for i, chunk_text in enumerate(chunks_text):
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": i,
            "chunk_index": i,
            "source": source,
            "filename": Path(source).name,
            "text": chunk_text,
            "char_count": len(chunk_text),
            "token_count": len(chunk_text) // 4,
            "chunk_time": datetime.now().isoformat()
        })
    
    return chunks


def chunk_file(input_path: str, output_path: str, method: str = 'paragraph',
               chunk_size: int = 512, overlap: int = 50) -> Dict[str, Any]:
    """切分单个 JSONL 文件"""
    chunks = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                doc_chunks = chunk_document(doc, method, chunk_size, overlap)
                chunks.extend(doc_chunks)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    return {
        "status": "success",
        "input_file": input_path,
        "output_file": output_path,
        "docs_processed": len(chunks),
        "chunks_total": len(chunks),
        "method": method,
        "chunk_size": chunk_size,
        "overlap": overlap
    }


def chunk_directory(input_dir: str, output_path: str, method: str = 'paragraph',
                    chunk_size: int = 512, overlap: int = 50) -> Dict[str, Any]:
    """切分目录中的所有 JSONL 文件"""
    all_chunks = []
    files_processed = 0
    
    for file_path in Path(input_dir).glob('*.jsonl'):
        result = chunk_file(str(file_path), output_path, method, chunk_size, overlap)
        files_processed += 1
        
        with open(output_path, 'r', encoding='utf-8') as f:
            all_chunks = [json.loads(line) for line in f if line.strip()]
    
    return {
        "status": "success",
        "input_dir": input_dir,
        "output_file": output_path,
        "files_processed": files_processed,
        "chunks_total": len(all_chunks),
        "method": method,
        "chunk_size": chunk_size,
        "overlap": overlap
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文本切分模块')
    parser.add_argument('--input', '-i', required=True, help='输入 JSONL 文件或目录')
    parser.add_argument('--output', '-o', default='./data/processed/docs_chunks.jsonl',
                        help='输出 JSONL 文件路径')
    parser.add_argument('--method', '-m', default='paragraph', 
                        choices=['paragraph', 'char', 'sentence'],
                        help='切分方法: paragraph(按段落), char(按字符), sentence(按句子)')
    parser.add_argument('--chunk-size', '-s', type=int, default=512,
                        help='块大小（字符数），仅 char 方法有效')
    parser.add_argument('--overlap', '-v', type=int, default=50,
                        help='重叠字符数，仅 char 方法有效')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = chunk_file(str(input_path), args.output, args.method, 
                           args.chunk_size, args.overlap)
    elif input_path.is_dir():
        result = chunk_directory(str(input_path), args.output, args.method,
                                args.chunk_size, args.overlap)
    else:
        print(f"错误: 路径不存在: {args.input}")
        return 1
    
    print(f"状态: {result['status']}")
    print(f"输入: {result.get('input_file', result.get('input_dir', 'N/A'))}")
    print(f"输出: {result['output_file']}")
    print(f"切分方法: {result['method']}")
    print(f"块大小: {result['chunk_size']} 字符")
    print(f"重叠: {result['overlap']} 字符")
    print(f"处理文件数: {result.get('files_processed', 1)}")
    print(f"总块数: {result['chunks_total']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
