#!/usr/bin/env python
# -*- coding: utf-8 -*- #

AUTHOR = 'toohsk'
SITENAME = "Atsushi's blog"
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Asia/Tokyo'

DEFAULT_LANG = 'ja'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),
         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('You can add links in your config file', '#'),
          ('Another social link', '#'),)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

# setting for themes/clean-blog
THEME = './themes/clean-blog'
HEADER_COVER = 'images/blog_cover.jpg'
COLOR_SCHEME_CSS = 'github.css'
GITHUB_URL = 'http://github.com/toohsk'
TWITTER_URL = 'http://twitter.com/toohsk'
SOCIAL = (('twitter', TWITTER_URL),
          ('github', GITHUB_URL))

