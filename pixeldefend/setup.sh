#!/bin/bash

cd "$(dirname "$0")" # cd to directory of this script

# $1 is filename
# $2 is expected sha
check_sha1() {
    computed=$(sha1sum "$1" 2>/dev/null | awk '{print $1}') || return 1
    if [ "$computed" == "$2" ]; then
        return 0;
    else
        return 1;
    fi
}

# $1 is URL
# $2 is extracted file name
# $3 is the checksum
fetch() {
    if check_sha1 $2 $3; then
        echo "$2 already downloaded"
        return
    fi
    echo "downloading $1"
    f=${1##*/}
    wget -q $1 -O $f
    unzip $f
    rm -f $f
    if check_sha1 $2 $3; then
        echo "downloaded $2"
    else
        echo "HASH MISMATCH, SHA1($2) != $3"
    fi
}

cd data

fetch \
    http://alpha.openai.com/pxpp.zip \
    params_cifar.ckpt c0913e0aea902ad33dde81c96b3d8cea8c0687e9

fetch \
    'https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1' \
    'models/naturally_trained/checkpoint-70000.data-00000-of-00001' 67373221b0a4324ffd7b8353160f9d885b166ed5
