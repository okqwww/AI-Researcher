#!/usr/bin/env python3
"""测试 arXiv 论文搜索功能"""

import sys
sys.path.insert(0, '/home/a/AI-Researcher/research_agent')

from inno.tools.arxiv_source import search_arxiv, download_arxiv_source_by_title

paper_list = [
    "Neural discrete representation learning",
    "Vector-quantized image modeling with improved vqgan",
    "High-resolution image synthesis with latent diffusion models",
    "Auto-encoding variational bayes",
    "Categorical reparameterization with gumbel-softmax"
]

print("=" * 60)
print("测试 search_arxiv 函数")
print("=" * 60)

for title in paper_list:
    print(f"\n搜索: '{title}'")
    print("-" * 40)
    
    try:
        papers = search_arxiv(title, max_results=3)
        print(f"找到 {len(papers)} 篇论文")
        
        if len(papers) > 0:
            for i, paper in enumerate(papers):
                print(f"\n  [{i+1}] {paper['title']}")
                print(f"      URL: {paper['url']}")
                print(f"      作者: {', '.join(paper['author'][:3])}...")
        else:
            print("  ❌ 未找到任何论文")
    except Exception as e:
        print(f"  ❌ 搜索出错: {e}")

print("\n" + "=" * 60)
print("测试 download_arxiv_source_by_title 函数")
print("=" * 60)

# 测试下载（使用临时目录）
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    result = download_arxiv_source_by_title(
        paper_list=paper_list,
        local_root=tmpdir,
        workplace_name="test_workplace"
    )
    print("\n下载结果:")
    print(result)
    
    # 检查是否有文件下载成功
    papers_dir = os.path.join(tmpdir, "test_workplace", "papers")
    if os.path.exists(papers_dir):
        files = os.listdir(papers_dir)
        print(f"\n下载的文件: {files}")
    else:
        print("\n❌ papers 目录不存在，下载失败")
