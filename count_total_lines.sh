#!/bin/bash 
echo "I'm gonna count up all the lines for you <3, give me two seconds"
sleep 2
echo
echo 'python'
git ls-files | grep -e '.py'|xargs wc -l

echo
echo
echo 'markdown'
git ls-files | grep -e '.md'|xargs wc -l

