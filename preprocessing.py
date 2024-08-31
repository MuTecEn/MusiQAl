# %%
import pandas as pd
import os
from pytube import YouTube
import subprocess
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

# %% Downloading YouTube videos

data_sources = pd.read_excel("your_path_to/audiovideo_sources.xlsx", sheet_name=None, header=None)

for ptype in [list(data_sources.keys())[0]]: # 0 for inst solo, 1 for inst duet, etc
    df=data_sources[ptype]
    df=df.iloc[0:,]
    output_path = f'the/path/to/your/output/folder/{ptype}'
    for i,track in enumerate(df.iterrows()):
        video_url = track[1][2]
        stamp_start=track[1][3]
        stamp_tmp=str((pd.to_datetime(str(stamp_start))+datetime.timedelta(hours=1)).time())
        stamp_tmp=':'.join(stamp_tmp.split(':')[:2])
        stamp_tmp2=str((pd.to_datetime(str(stamp_start))).time())
        stamp_start=':'.join(stamp_tmp2.split(':')[:2])
        stamp_stop=track[1][4]
        if not pd.isna(stamp_stop):
            stamp_tmp3=str((pd.to_datetime(str(stamp_stop))).time())
            stamp_tmp3=':'.join(stamp_tmp3.split(':')[:2])
        stamp_stop = stamp_tmp if pd.isna(track[1][4]) else stamp_tmp3
        print(f'{track[1]},from {stamp_start}-{stamp_stop}')
        if 'youtube' in video_url:
            subprocess.check_call(f'yt-dlp --force-overwrites -S vcodec:h264,res,acodec:m4a --download-sections "*{stamp_start}-{stamp_stop}" -o "{output_path}/{i}.%(ext)s" {video_url}', shell=True)
        else:
            subprocess.check_call(f'ffmpeg -y -ss {stamp_start} -to {stamp_stop} -i {video_url} -c copy "{output_path}/{i}.mp4"', shell=False)            # 

# %%
# read all sheets of collected_qa_pairs
data_sources = pd.read_excel("path/to/collected_qa_pairs.xlsx", sheet_name=None) # if you are working on your own dataset, replace this file

# %% 

# Check if the answer of the question is empty, then ignore the question

list_dfs=[]
for sheet in data_sources:
        data_sources[sheet]=data_sources[sheet].dropna(subset='answer')
        data_sources[sheet]['sheet']=sheet
        list_dfs.append(data_sources[sheet])

df_concat=pd.concat(list_dfs)

df_concat['answer']=df_concat['answer'].str.lower()

question_unique=df_concat['question_content'].str.replace('? ', '?').sort_values().unique()

# print(f'{question_unique.shape[0]} unique questions')
# question_unique
# %%
# Check how often a unique question occurs

question_count = df_concat['question_content'].str.replace('? ', '?').value_counts()

question_count
plt.subplots(figsize=(30,10))
sns.barplot(x=question_count.index, y=question_count)
plt.xticks(rotation=45, ha='right')

# %%

# Give IDs to questions

video_ids=pd.Series(df_concat_del['video_id'].unique()).dropna().to_list()
video_ids_bool=df_concat_del.video_id.isin(video_ids)
assert len(video_ids)==video_ids_bool.sum()
df_concat_del.video_id.value_counts()

video_ids
video_ids_new= ["{:08d}".format(i) for i in list(range(0,len(video_ids)))]
# make a dictionary for the values (old and new)
id_map = dict(zip(video_ids,video_ids_new))

video_ids_bool
video_ids_fixed=[]
video_ids_fixed_new = []
id_i=-1
for id_bool in video_ids_bool:
    if id_bool:
        id_i+=1
    video_ids_fixed.append(video_ids[id_i])
    video_ids_fixed_new.append(video_ids_new[id_i])

assert len(video_ids_fixed)==df_concat_del.shape[0]

df_concat_del['video_id_fixed']=video_ids_fixed
df_concat_del['video_id_fixed_new']=video_ids_fixed_new

# %%

# Create a common folder for all videos, giving them unique names 

# JSON file
import shutil
# output folder for videos
video_input = "your/data/folder"
video_output=r'your/new/data/folder'
os.makedirs(video_output, exist_ok=True)

# Loop through the dictionary to rename videos
for id,id_new in id_map.items():
    folder = f'{video_input}/{"_".join(id.split("_")[:-1])}'
    filename = id.split("_")[-1]
    filename_new = id_new.split("_")[-1]
    print(id_new)

    shutil.copy(f'{folder}/{filename}.mp4', f'{video_output}/{filename_new}.mp4')

# %%

# Temporary json file

df_json=df_concat_del.copy()
df_json['video_id']=df_json['video_id_fixed_new']
df_json=df_json.drop(columns=['video_id_fixed','video_id_fixed_new','index','Unnamed: 6','Unnamed: 8','HELP','sheet'])

col_names=df_json.columns.to_list()
new_df = df_json[[col_names[0],col_names[1],col_names[2],col_names[3],col_names[4],col_names[6],col_names[5]]]

df_json.to_json('file.json', orient = 'records',indent=3) #, compression = 'infer', index = 'true')


# %%

# replace with your questions
questions_dict = {
    "Counting": {
        "Is there a <Object> sound?": "Audio",
        "Are there <Object> and <Object> sound?": "Audio",
        "How many musical instruments were heard throughout the video?": "Audio",
        
        "Is there a <Object> in the entire video?": "Visual",
        "Are there <Object> and <Object> instruments in the video?": "Visual",
        "How many types of musical instruments appeared in the entire video?": "Visual",
        "How many <Object> are in the entire video?": "Visual",

        "How many sounding <Object> in the video?": "Audio-Visual",
        "How many instruments in the video did not sound from beginning to end?": "Audio-Visual",
        "How many types of musical instruments sound in the video?": "Audio-Visual",
        "How many instruments are sounding in the video?": "Audio-Visual",
        
        "Are there any lyrics?": "Audio",
        "Are there any vocals?": "Audio",
        "How many musical instruments were heard throughout the video?": "Audio-Visual",
        "How many performers are dancing in the video?": "Visual",
        "How many performers are playing instruments in the video?": "Audio-Visual",
        "How many performers are singing in the video?": "Audio-Visual",
        "How many singing voices were heard throughout the video?": "Audio-Visual",
        "How many solos are featured in the music video?": "Audio-Visual",
        "How many times does the performer visually interact with the audience?": "Visual",
        "How many times does the tempo change?": "Audio",
        "How many voices in the video did not sound from beginning to end?": "Audio-Visual",
    },

    "Existential": {
        "Is there an offscreen sound?": "Audio-Visual",
        "Is the <Object> in the video always playing?": "Audio-Visual",
        "Is this sound from the instrument in the video?": "Audio-Visual",
        
        "Did the <object> gestures convey the emotion of the music?": "Audio-Visual",
        "Is the dancer in the video always dancing?": "Visual",
        "Is the singer in the video always singing?": "Audio-Visual",
        "Is the sound of the video being played live or pre-recorded?": "Audio-Visual",
        "Is there any background ambiance or environmental sounds audible during the music performance?": "Audio",
        "Is this sound from the voice in the video?": "Audio-Visual",
        "What kind of dance is it?": "Audio-Visual",
        "What kind of music genre is it?": "Audio-Visual",
        "What kind of singing style is it?": "Audio",
        "What type of performance is happening in this video?": "Audio-Visual",
    },

    "Location": {
        "Where is the performance?": "Visual",
        "What is the instrument on the <LR> of <Object>?": "Visual",
        "What kind of musical instrument is it?": "Visual",
        "What kind of instrument is the <LRer> instrument?": "Visual",

        "Where is the <LL> instrument?": "Audio-Visual",
        "Is the <FL> sound coming from the <LR> instrument?": "Audio-Visual",
        "What is the <LR> instrument of the <FL> sounding instrument?": "Audio-Visual",

        "From which direction or location is the sound source coming?": "Audio-Visual",
        "Is there distance between <Object> and <Object>?": "Visual",
        "What is the spatial arrangement of the dancers on stage?": "Visual",
        "What is the spatial arrangement of the singers on stage?": "Visual",
        "What is the spatial arrangement of the instruments on stage?": "Visual",
        "Where is the audience in relation to the performance setup?": "Visual",
        "Where is the music coming from in this performance?": "Audio-Visual",
        "Where on stage is the dancer during the solo?": "Audio-Visual",
        "Where on stage is the musician when the solo is played?": "Audio-Visual",
        "Where on stage is the singer during the solo?": "Audio-Visual",
    },

    "Temporal": {
        "Where is the <FL> sounding instrument?": "Audio-Visual",
        "Which instrument makes the sound <FL>?": "Audio-Visual",
        "Which instrument makes sounds <BA> the <Object>?": "Audio-Visual",
        "What is the <TH> instrument that comes in?": "Audio-Visual",

        "Did any performers change their positions on the stage during the music performance?": "Visual",
        "Did the <object> perform any visual actions during the video?": "Visual",
        "Did the visual theme of the stage change during the piece?": "Audio-Visual",
        "Does the energy level of the performance increase or decrease as it progresses?": "Audio-Visual",
        "How did the lighting change during the <Object>?": "Audio-Visual",
        "Which is the musical instrument that sounds at the same time as the <Object>?": "Audio",
    },

    "Comparative": {
        "Is the <Object> more rhythmic than the <Object>?": "Audio",
        "Is the <Object> louder than the <Object>?": "Audio",
        "Is the <Object> playing longer than the <Object>?": "Audio",

        "Is the instrument on the <LR> louder than the instrument on the <LR>?": "Audio-Visual",
        "Is the instrument on the <LR> more rhythmic than the instrument on the <LR>?": "Audio-Visual",
        "Is the <Object> on the <LR> louder than the <Object> on the <LR>?": "Audio-Visual",
        "Is the <Object> on the <LR> more rhythmic than the <Object> on the <LR>?": "Audio-Visual",
        
        "Did the <object> gestures correspond with the <object> fills?": "Audio-Visual",
        "Do the performers visual cues coordinate with the rhythm of the music?": "Audio-Visual",
        "Does <Object> play the same rhythm with <Object>?": "Audio",
    },

    "Causal": {
        "Does the venue's acoustics affect instrument placement and sound projection?": "Audio-Visual",
        "Why are the performers wearing this attire?": "Visual",
        "Why did the audience cheer?": "Audio-Visual",
        "Why did the audience clap?": "Audio-Visual",
        "Why did the performer use <Object>?": "Audio-Visual",
        "Why does the dancer pause during the song?": "Audio-Visual",
        "Why does the singer pause during the song?": "Audio-Visual",
        "Why is there a spoken word segment in the middle of the song?": "Audio",
    },

    "Purpose": {
        "Was the use of props to add visual interest or to convey a specific theme?": "Audio-Visual",
        "What is the purpose of <Object> in the video?": "Audio-Visual",
        "What is the purpose of the sound in the video?": "Audio-Visual",
    }
}

# %%

# Make a json file for all your data, mapping templ_values to the correct answers

df  = new_df

# Define a function to determine the question types
def determine_question_type(question_content):
    for category, questions in questions_dict.items():
        for ques, q_type in questions.items():
            if ques in question_content:
                return f'["{q_type}", "{category}"]'
    return '["Unknown"]'  # Default type if not found in the dictionary

# Apply the function to the DataFrame
df['type'] = df['question_content'].apply(determine_question_type)

# Update the question_id to be a count starting from 1
df['question_id'] = range(1, len(df) + 1)

# Function to format templ_values
def format_templ_values(values):
    if pd.isna(values) or values == "":
        return '[]'
    
    if isinstance(values, str):
        # Assuming values are separated by commas
        values_list = [v.strip() for v in values.split(",")]
    elif isinstance(values, list):
        values_list = values
    else:
        values_list = [values]
    
    # Convert to a JSON string
    json_string = json.dumps(values_list)
    return json_string

# Apply the formatting function to templ_values with awareness of df
df['templ_values'] = df['templ_values'].apply(format_templ_values)

# Ensure all question_deleted are 0
df['question_deleted'] = 0

# Display the updated DataFrame
print(df.head())

df = df.sample(frac=1).reset_index(drop=True)

# Save the updated DataFrame to a new JSON file
df.to_json('file.json', orient='records', indent=3)

# %% Stratified Sampling for Train, Val, Test

from sklearn.model_selection import train_test_split

# Assuming 'label' is the column you want to stratify by
y = df_bal['answer']

# Split into train (80%) and test (20%) while stratifying
train_val, test = train_test_split(df_bal, test_size=0.2, random_state=3, stratify=y)

# Further split train_val into train (70%) and validate (10%) while stratifying
y_train_val = train_val['answer']
train, validate = train_test_split(train_val, test_size=0.125, random_state=2, stratify=y_train_val)  # 0.125 * 80% = 10%

# Print the shape ratios for verification
print(f'Train: {train.shape[0] / df_bal.shape[0]}')
print(f'Validate: {validate.shape[0] / df_bal.shape[0]}')
print(f'Test: {test.shape[0] / df_bal.shape[0]}')

# Save the splits to JSON files
train.to_json('avqa-train.json', orient='records', indent=3)
validate.to_json('avqa-val.json', orient='records', indent=3)
test.to_json('avqa-test.json', orient='records', indent=3)

