#!/bin/bash

set -o errexit -o nounset

if [ "$TRAVIS_BRANCH" != "master" ]
then
  echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
  exit 0
fi
if [ -z "${GH_TOKEN}" ] ; then
  echo "GH_TOKEN is not set. Exiting..."
  exit 1
fi
[ "$TRAVIS_RUST_VERSION" = "stable" ] || exit 0
[ "$TRAVIS_PULL_REQUEST" = false ] || exit 0

rev=$(git rev-parse --short HEAD)

RELEASE="master"
CRATE_NAME="lockfreehashmap"
COMMIT_COMMENT="Update developer docs for commit ${rev}"
echo "Publishing documentation for $RELEASE"

if [ ! -d "target/doc/$CRATE_NAME" ]; then
  echo "Cannot find target/doc/$CRATE_NAME"
  exit 1
fi

git clone --depth=50 --branch=gh-pages "https://github.com/${TRAVIS_REPO_SLUG}.git" gh-pages
rm -Rf "gh-pages/${RELEASE}"
mkdir "gh-pages/${RELEASE}"

echo "<meta http-equiv=refresh content=0;url=${CRATE_NAME}/index.html>" > target/doc/index.html
cp -R target/doc/* "gh-pages/${RELEASE}/"

INDEX="gh-pages/index.html"
echo "<html><head><title>${TRAVIS_REPO_SLUG} API documentation</title></head><body>" > $INDEX
echo "<h1>API documentation for crate <a href='https://github.com/${TRAVIS_REPO_SLUG}'>${CRATE_NAME}</a></h1>" >> $INDEX
echo "<strong>Select a crate version :</strong><br/>" >> $INDEX
for entry in $(ls -1 gh-pages); do
  [ -d "gh-pages/$entry" ] && echo "<a href='${entry}/index.html'>${entry}</a><br/>" >> $INDEX
done
echo "</body></html>" >> $INDEX

cd gh-pages || exit 1
git config user.name "travis-ci"
git config user.email "travis@travis-ci.org"

git add --all
git commit -m "$COMMIT_COMMENT"

git push -fq "https://${GH_TOKEN}@github.com/${TRAVIS_REPO_SLUG}.git" gh-pages || exit 1
cd ..
rm -Rf "gh-pages"
