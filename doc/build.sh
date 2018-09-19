make clean -e SPHINXOPTS="-D language='ja'" html
python rewrite-tutorial.py ja
mv _build/html _build/ja

make -e SPHINXOPTS="-D language='en'" html
python rewrite-tutorial.py en
mv _build/ja _build/html/ja