today=`date -u "+%Y-%m-%d"`
cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ../ai
python enhance.py --data ../data/${today}.jsonl

cd ../to_md
python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl

# 新增下面这一行，确保后续命令在项目根目录执行
cd ..

ls data/*.jsonl | sed 's|data/||' > assets/file-list.txt
