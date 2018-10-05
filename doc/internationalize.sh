#!/bin/bash


CMDNAME=`basename $0`


while getopts ul OPT
do
  case $OPT in
    "u" ) FLG_A="TRUE" ;;
    "l" ) FLG_B="TRUE" ;;
      * ) echo "Usage: $CMDNAME [-u] [-l]" 1>&2
          exit 1 ;;
  esac
done

if [ "$FLG_A" = "TRUE" ]; then
    echo "Update po files."
    make gettext
    sphinx-intl update -p _build/locale -l ja
fi

if [ "$FLG_B" = "TRUE" ]; then
    make -e SPHINXOPTS="-D language='de'" html
fi

exit 0
