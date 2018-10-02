make clean -e SPHINXOPTS="-D language='ja'" html
mv _build/html _build/ja

make -e SPHINXOPTS="-D language='en'" html
mv _build/ja _build/html/ja
