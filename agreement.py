import json
import glob
import errno

buzz_fake_news_path = 'News_Data\\FakeNewsNet-master\\Data\\BuzzFeed\\FakeNewsContent\\*.json'

def get_json_title(json_file):
    json_string = json_file.read()
    json_dict = json.loads(json_string)
    return json_dict["title"]

def politi_buzz_analysis(path):
    files = glob.glob(path)
    title_list = []
    for file_name in files:
        try:
            with open(file_name, 'r') as json_file:
                article_title = get_json_title(json_file)
                title_list = title_list + [article_title]   
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    return title_list

art_list = politi_buzz_analysis(buzz_fake_news_path)

print(len(art_list))
