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
    tar xf $f
    rm -f $f
    if check_sha1 $2 $3; then
        echo "downloaded $2"
    else
        echo "HASH MISMATCH, SHA1($2) != $3"
    fi
}

cd data

fetch \
    http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz \
    inception_v3.ckpt 606fd6953c58c817c56fd3bc2f0384fc2ecaba92

fetch \
    https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/quilt_db.tar.gz \
    quilt_db.npy ff4b94f45c9e8441b341fd5ffcf2adb0a8049873
