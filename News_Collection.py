'''
    PROJECT STATEMENT:
        To create a ML pipeline that is able to classify headlines as
        coming from Left, Right or Center sources in the political spectrum.

    FILE STATEMENT:
        This file performs the following processes:
            1. Captures headlines from websites, labeled as right, left or center
            2. Saves headlines w/ respective label as a csv file
'''

# Inport Libraries
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
from datetime import datetime
import os


########## OBTAIN HEADLINES ##########


# Define Function to get articles headline and organize into dataframe
def get_allsides_titles():
    '''Parses through left, center and right-inclined headlines
    Source: www.allsides.com
    Returns a dataframe with headings classified as left, center, right
    '''

    # Provide URLs
    main_url = 'https://www.allsides.com/unbiased-balanced-news'

    topics_list = ['bridging-divides', 'criminal-justice', 'election-2012', 'environment',
                   'healthcare', 'gay-rights', 'nuclear-weapons', 'taxes', 'campaign-finance', 'economy-jobs',
                   'elections', 'free-speech', 'immigration', 'media-bias-media-watch', 'polarization', 'terrorism',
                   'civil-rights', 'education', 'energy', 'guns', 'inequality',
                   'middle-east', 'role-government', 'trade']
    url_list = ["https://www.allsides.com/topics/{}".format(topic) for topic in topics_list]
    url_list.append(main_url)

    # Create main data frame where all headlines will be added
    df = pd.DataFrame(columns=['Headlines', 'Source', 'Inclination'])

    # Loop through URLs to get news titles
    for url in url_list:
        #         print('Getting Data From: {}'.format(url))
        html = urlopen(url)

        # Read all html data
        soup = BeautifulSoup(html.read(), "html.parser")

        # Get headlines and add to dataframe
        for side in ['left', 'right', 'center']:
            section = soup.find(class_='region-triptych-{}'.format(side))
            title_links = section.find_all(class_='news-title')
            source_links = section.find_all(class_='news-source')
            titles = [i.get_text()[1:-1] for i in title_links]
            sources = [i.get_text() for i in source_links]

            df_side = pd.DataFrame({'Headlines': titles,
                                    'Inclination': [side] * len(titles),
                                    'Source': sources})

            df = pd.concat([df, df_side], axis=0, sort=True)

    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(inplace=True)

    return (df)


# Get Headlines
allsides_df = get_allsides_titles()

# Save Today's DataFrame
date = datetime.today().strftime('%Y-%m-%d')
this_dir = os.path.dirname(os.path.abspath("__file__"))
dump_dir = os.path.join(this_dir, "Extracted Data")
file_path = os.path.join(dump_dir, "allsides_{}{}".format(date,".csv"))
allsides_df.to_csv(file_path)