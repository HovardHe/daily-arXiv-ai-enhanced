today=`date -u "+%Y-%m-%d"`
cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ../ai
python enhance.py --data ../data/${today}.jsonl

cd ../to_md
python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl

# 确保切换回根目录
cd ..

ls data/*.jsonl | sed 's|data/||' > assets/file-list.txt
