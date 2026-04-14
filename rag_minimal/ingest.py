#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档导入模块
读取本地文档，支持 txt、md、pdf，输出统一格式 JSONL
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


def extract_text_from_txt(file_path: str) -> str:
    """从 txt 文件提取文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()


def extract_text_from_md(file_path: str) -> str:
    """从 md 文件提取文本"""
    return extract_text_from_txt(file_path)


def extract_text_from_pdf(file_path: str) -> str:
    """从 pdf 文件提取文本"""
    if PyPDF2 is None:
        raise ImportError("PyPDF2 未安装，请运行: pip install PyPDF2")
    
    text = []
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    
    return '\n'.join(text)


def extract_text(file_path: str) -> str:
    """根据文件扩展名提取文本"""
    ext = Path(file_path).suffix.lower()
    
    if ext == '.txt':
        return extract_text_from_txt(file_path)
    elif ext == '.md':
        return extract_text_from_md(file_path)
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


def scan_directory(input_dir: str, extensions: List[str] = None) -> List[str]:
    """扫描目录中的文档文件"""
    if extensions is None:
        extensions = ['.txt', '.md', '.pdf']
    
    files = []
    input_path = Path(input_dir)
    
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            files.append(str(file_path))
    
    return sorted(files)


def ingest_document(file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
    """导入单个文档"""
    if doc_id is None:
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(file_path) % 10000:04d}"
    
    file_path = Path(file_path)
    text = extract_text(str(file_path))
    
    return {
        "doc_id": doc_id,
        "source": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "text": text,
        "char_count": len(text),
        "ingest_time": datetime.now().isoformat()
    }


def ingest_directory(input_dir: str, output_path: str) -> Dict[str, Any]:
    """导入目录中的所有文档"""
    files = scan_directory(input_dir)
    
    if not files:
        return {
            "status": "warning",
            "message": f"目录 {input_dir} 中没有找到文档文件",
            "files_processed": 0,
            "output_path": output_path
        }
    
    docs = []
    for file_path in files:
        try:
            doc = ingest_document(file_path)
            docs.append(doc)
        except Exception as e:
            print(f"处理文件失败 {file_path}: {e}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    return {
        "status": "success",
        "message": f"成功导入 {len(docs)} 个文档",
        "files_processed": len(docs),
        "output_path": output_path,
        "docs": docs
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='文档导入模块')
    parser.add_argument('--input', '-i', required=True, help='输入文件或目录路径')
    parser.add_argument('--output', '-o', default='./data/processed/docs_raw.jsonl', 
                        help='输出 JSONL 文件路径')
    parser.add_argument('--doc-id', '-d', default=None, help='文档ID（仅单文件时有效）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        doc = ingest_document(str(input_path), args.doc_id)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"成功导入文档: {doc['filename']}")
        print(f"字符数: {doc['char_count']}")
        print(f"输出文件: {args.output}")
        
    elif input_path.is_dir():
        result = ingest_directory(str(input_path), args.output)
        
        print(f"状态: {result['status']}")
        print(f"消息: {result['message']}")
        print(f"处理文件数: {result['files_processed']}")
        print(f"输出文件: {result['output_path']}")
        
        if result['status'] == 'success':
            for doc in result['docs'][:3]:
                print(f"  - {doc['filename']} ({doc['char_count']} 字符)")
            if len(result['docs']) > 3:
                print(f"  ... 还有 {len(result['docs']) - 3} 个文档")
    else:
        print(f"错误: 路径不存在: {args.input}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
