echo 'python'
git ls-files | grep -e '.py'|xargs wc -l

echo
echo
echo 'markdown'
git ls-files | grep -e '.md'|xargs wc -l
