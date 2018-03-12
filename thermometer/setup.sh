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

cd ..
fetch https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/cifar10_data.tgz cifar10_data 6d011cbb029aec2c18dc10bce32adea9e27c2068
mkdir -p models
cd models
fetch https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/model_thermometer_advtrain.tgz models/thermometer_advtrain 595261189ee9a78911f312cd2443ee088ef59bee
