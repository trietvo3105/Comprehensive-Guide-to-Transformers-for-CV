#!/bin/bash

echo "Installing dependencies..."
bundle install

echo "Building site..."
bundle exec jekyll build

echo "Site built successfully! Check the _site directory for the generated files."
echo "To preview the site locally, run: bundle exec jekyll serve" 