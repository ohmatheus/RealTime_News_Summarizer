import os  # helper functions like check file exists
import datetime  # automatic file name
import requests  # the following imports are common web scraping bundle
from urllib.request import urlopen  # standard python module
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from collections import defaultdict
import re
from urllib.error import URLError
from tqdm import tqdm
import pickle
import bz2
import pandas as pd
from collections import Counter
from urllib.parse import urlparse, unquote
import json
import shutil


#-----------------------------------------------------
def extract_theme(link):
    try:
        theme_text = re.findall(r'.fr/.*?/', link)[0]
    except:
        pass
    else:
        return theme_text[4:-1]


#-----------------------------------------------------
def extract_name_from_https_link(link):
    """
    Extracts the domain name (name) from an HTTPS link.

    :param link: A string containing the HTTPS link.
    :return: The domain name (e.g., 'www.google.com') or None if no match is found.
    """
    # Regular expression to match the domain name in an HTTPS link
    match = re.search(r'https://([^/\s]+)', link)
    if match:
        return match.group(1)  # Return the domain name
    return None  # Return None if no match is found


#-----------------------------------------------------
def extract_subname(url):
    """Extracts the subname (article slug) from a given HTML link."""
    path = urlparse(url).path  # Extract the path from the URL
    filename = os.path.basename(path)  # Get the last part of the path
    subname = os.path.splitext(filename)[0]  # Remove the .html extension
    return unquote(subname)  # Decode any URL-encoded characters


#-----------------------------------------------------
def extract_clean_subname(url):
    subname = extract_subname(url)
    return re.sub(r'(_\d+)+$', '', subname)  # Remove all trailing underscores followed by numbers


#-----------------------------------------------------
def get_filename(filepath):
    """Extracts the filename from a given file path."""
    return os.path.basename(filepath)


#-----------------------------------------------------
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"


#-----------------------------------------------------
def list_themes(links):
    themes = []
    for link in links:
        theme = extract_theme(link)
        if theme is not None:
            themes.append(theme)
    return themes


#-----------------------------------------------------
def write_links(path, links, year_fn):
    with open(os.path.join(path + "/lemonde_" + str(year_fn) + "_links.txt"), 'w', encoding="utf-8") as f:
        for link in links:
            f.write(link + "\n")


#-----------------------------------------------------
def write_to_file(filename, content):
    if os.path.exists(filename):
        with open(filename, 'a+', encoding="utf-8") as f:
            f.write(str(content))
    else:
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(str(content))


#-----------------------------------------------------
def create_archive_links(year_start, year_end, month_start, month_end, day_start, day_end):
    archive_links = {}
    for y in range(year_start, year_end + 1):
        dates = [str(d).zfill(2) + "-" + str(m).zfill(2) + "-" +
                    str(y) for m in range(month_start, month_end + 1) for d in
                    range(day_start, day_end + 1)]
        archive_links[y] = [
            "https://www.lemonde.fr/archives-du-monde/" + date + "/" for date in dates]
    return archive_links


#-----------------------------------------------------
def get_articles_links(archive_links):
    links_non_abonne = []
    for link in archive_links:
        try:
            html = urlopen(link)
        except HTTPError as e:
            print("url not valid", link)
        else:
            soup = BeautifulSoup(html, "html.parser")
            news = soup.find_all(class_="teaser")
            # condition here : if no span icon__premium (abonnes)
            for item in news:
                if not item.find('span', {'class': 'icon__premium'}):
                    l_article = item.find('a')['href']
                    # en-direct = video
                    if 'en-direct' not in l_article:
                        links_non_abonne.append(l_article)
    return links_non_abonne


#-----------------------------------------------------
def classify_links(theme_list, link_list):
    dict_links = defaultdict(list)
    for theme in theme_list:
        theme_link = 'https://www.lemonde.fr/' + theme + '/article/'
        for link in link_list:
            if theme_link in link:
                dict_links[theme].append(link)
    return dict_links


#-----------------------------------------------------
def get_single_page(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        print("url not valid", url)
    else:
        soup = BeautifulSoup(html, "html.parser")
        text_title = soup.find('h1')
        text_body = soup.article.find_all(["p", "h2"], recursive=False)
        return (text_title, text_body)


#-----------------------------------------------------
def scrape_articles(dict_links):
    themes = dict_links.keys()
    for theme in themes:
        create_folder(os.path.join('corpus', theme))
        print("processing:", theme)
        for i in tqdm(range(len(dict_links[theme]))):
            link = dict_links[theme][i]
            fn = extract_clean_subname(link)
            single_page = get_single_page(link)
            # Add metadata with link, title, eventually tags
            title = single_page[0].get_text()
            #formatted_title = title.encode('utf-8').decode('unicode_escape')
            formatted_title = title
            data = {
                "link": f"{link}",
                "title": f"{formatted_title}"
                }
            if single_page is not None:
                with open((os.path.join('corpus', theme, fn + '.txt')), 'w', encoding="utf-8") as f:
                    # f.write(dict_links[theme][i] + "\n" * 2)
                    f.write(single_page[0].get_text() + "\n")
                    for line in single_page[1]:
                        f.write(line.get_text() + "\n")
                with open((os.path.join('corpus', theme, fn + '.meta')), 'w', encoding="utf-8") as f:
                    json.dump(data, f)


#-----------------------------------------------------
def cr_corpus_dict(path_corpus, n_files=1000):
    dict_corpus = defaultdict(list)
    themes = os.listdir(path_corpus)
    for theme in themes:
        counter = 0
        if not theme.startswith('.'):
            theme_directory = os.path.join(path_corpus, theme)
            for file in os.listdir(theme_directory):
                if counter < n_files:
                    path_file = os.path.join(theme_directory, file)
                    text = read_file(path_file)
                    dict_corpus["label"].append(theme)
                    dict_corpus["text"].append(text)
                counter += 1
    return dict_corpus


#-----------------------------------------------------
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print("folder exists already")

#-----------------------------------------------------
def erase_folder_contents(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")