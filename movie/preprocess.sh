#!/bin/bash

system=`awk -F= '/^NAME/{print $2}' /etc/os-release`
echo ${system}
case "${system}" in
    "\"CentOS Linux\"")
            echo "CentOS System"
            sudo yum -y install unzip
            ;;
    \""Ubuntu\"")
            echo "Ubuntu System"
            sudo apt-get -y install unzip
            ;;
    *)
            echo "Not support this system."
esac
echo "Installed unzip"

wget -nc http://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip