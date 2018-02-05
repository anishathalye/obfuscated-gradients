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
    f=${1##*/}
    if check_sha1 $f $3; then
        echo "$2 already downloaded"
	return
    fi
    echo "downloading $1"
    wget -q $1 -O $f
    if check_sha1 $f $3; then
        echo "downloaded $2"
    else
        echo "HASH MISMATCH, SHA1($f) != $3"
	return
    fi

    tar xzf $f
}

fetch https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/defensegan_data.tgz data 00a0eeeae06896ae51aa705985572e06e7119c61
