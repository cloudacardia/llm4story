import pickle
import json
import random
import os, re
import torch
from sentence_transformers import SentenceTransformer, util
from utils import *

with open("key.txt", 'r', encoding='utf-8') as f:
    keys = [i.strip() for i in f.readlines()]

def multi_corpus_recommend(entry_name,query,corpus,k=4):
    if not os.path.exists('../data/'+entry_name+'.pkl'):
        first = True
        corpus_embeddings = {}
    else:
        first = False
        with open('../data/'+entry_name+'.pkl', 'rb') as f:
            corpus_embeddings = pickle.load(f)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='../models/sbert')
    query_embedding = model.encode(query)
    hits = []
    for i, item in enumerate(corpus):
        if item == []:
            hits.append({'corpus_id': i, 'score': 0})
            continue

        if first:
            item_embeddings = model.encode(item)
            corpus_embeddings[str(i)] = item_embeddings
        else:
            item_embeddings = corpus_embeddings[str(i)]

        similarity_sum = 0
        for sentence_embedding in item_embeddings:
            similarity = util.cos_sim(sentence_embedding, query_embedding).mean(-1)
            similarity_sum += similarity
        similarity_sum/=len(item)
        hits.append({'corpus_id': i, 'score': similarity_sum})
    if first:
        file = open('../data/'+entry_name+'.pkl', 'wb')
        pickle.dump(corpus_embeddings, file)
        file.close()

    hits = sorted(hits, key=lambda x: x['score'], reverse=True)[:k]
    recommendation = [(i['corpus_id'],corpus[i['corpus_id']]) for i in hits]
    return recommendation


def single_corpus_recommend(entry_name,query,corpus,k=4):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder='../models/sbert')
    if not os.path.exists('../data/'+entry_name+'.bin'):
        corpus_embeddings = model.encode(corpus)
        torch.save(corpus_embeddings,'../data/'+entry_name+'.bin')
    else:
        corpus_embeddings = torch.load('../data/'+entry_name+'.bin')
    query_embeddings = model.encode(query)

    # Find the top-2 corpus documents matching each query
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=k)
    #print(hits)
    recommendation = [(i['corpus_id'],corpus[i['corpus_id']]) for i in hits[0]]
    return recommendation

def clean_split(text):
    sentences = []
    if text is None:
        return sentences
    text = text.split('\n')
    for line in text:
        line = re.sub('\n','',line)
        line = re.sub('[1-9]*\.','',line)
        line = re.sub('\(*[1-9]*\)','',line)
        line = line.replace("- ","")
        line = line.replace("  "," ").strip()
        if line!="":
            sentences.append(line)
    return sentences

def get_all_recommend(movie_data, queries,topk=4):
    all_recommendations = {}

    all_styles, all_moods, all_subjects, all_genres, all_plots = [], [], [], [], []
    for i in range(len(movie_data)):
        mood = 'None' if movie_data[i]['mood'] is None else movie_data[i]['mood']
        style = 'None' if movie_data[i]['style'] is None else movie_data[i]['style']
        all_styles.append(style)
        all_moods.append(mood)

        all_genres.append(movie_data[i]['genre'])

        # if movie_data[i]['plot_summary'] == '':
        #     plots = movie_data[i]['plots']
        # else:
        #     plots = clean_split(movie_data[i]['plots'])

        plots = clean_split(movie_data[i]['plots'])
        subjects = clean_split(movie_data[i]['subjects'])

        all_subjects.append(subjects)
        all_plots.append(plots)

    #all_recommendations['mood_recommend'] = single_corpus_recommend('mood',queries['mood'], all_moods, k=topk)
    #all_recommendations['style_recommend'] = single_corpus_recommend('style',queries['style'], all_styles, k=topk)

    #all_recommendations['genre_recommend'] = multi_corpus_recommend('genre',queries['genre'], all_genres, k=topk)
    all_recommendations['subject_recommend'] = multi_corpus_recommend('subjects',queries['subjects'], all_subjects, k=topk)
    all_recommendations['plot_recommend'] = multi_corpus_recommend('plots',queries['plots'], all_plots, k=topk)

    return all_recommendations

def ask_why(story,new_info=[],plots=None,depth=0):

    def summarize(story):
        prompt = 'Here is a story: "' + story +'"\n' + 'Please summarize the above story and only give me an organized list of sentences, each of which describes one plot.\n'
        story_summary = generate(prompt)
        return story_summary

    if plots == None:
        story_summary = summarize(story)
        plots = clean_split(story_summary)

    new_plots = []
    for plot in plots:
        prompt = 'Here is a story: \n' + story +'\n' + 'Except for pure coincidence, give me a list of ideas about why does this happen: \n' + plot
        explanations = generate(prompt)
        explanations = clean_split(explanations)
        explanation_num = random.randint(0, len(explanations)-1)
        new_plots.append(explanations[explanation_num]) # pick an explanation randomly
        new_info.append((plot,explanations[explanation_num]))

    if depth < 1:
        depth += 1
        new_info = ask_why(story, new_info, new_plots, depth)
        return new_info
    return new_info

def add_new_info(story,new_info):
    num_parts = 2
    new_story = []
    reasons = [each[1] for each in new_info]
    reasons = "\n".join(reasons)
    prompt = 'Here is a story: \n' + story +'\n' +'Here is some additional information: \n'+ reasons + '\n'+ \
             'Please add the above information to the story and preserve other details by repeating the unchanged parts and only modifying the necessary paragraphs. As the story is too long, to make sure all the details are well covered, only describe the first half of the story and only tell the second half when I request.'
    messages = [{"role": "user", "content": prompt}]
    for part in range(num_parts):
        index_num = random.randint(0, len(keys) - 1)
        openai.api_key = keys[index_num]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        generation = response['choices'][0]['message']['content']

        new_story.append(generation)
        question = 'Describe the second half of the story.' #分两步生成故事
        messages.append({"role": "assistant", "content": generation})
        messages.append({"role": "user", "content": question})

    new_story = "\n".join(new_story)
    return new_story

if __name__ == '__main__':
    queries = {'mood':'excited, thrilled','style':'magical','genre':['action'],'subjects':['adventure','death'],'plots':['he escaped from the island','he met a mermaid','he had a soup']}

    with open('../data/movie_data.json', 'r') as f:
        movie_data = json.load(f)

    all_recommendations = get_all_recommend(movie_data, queries,4)

    example_id_1 = all_recommendations['plot_recommend'][0][0]
    example_id_2 = all_recommendations['plot_recommend'][1][0]

    example_1 = movie_data[example_id_1]
    example_2 = movie_data[example_id_2]
    examples = [example_1]
    #examples = []
    #examples = [example_1, example_2]

    prompt = make_prompt(examples=examples, conditions=queries)
    print(prompt)
    print("\n\n-----------------------------------------------\n\n")

    story = generate(prompt).replace("\n\n","\n")
    print(story)
    print("\n\n-----------------------------------------------\n\n")


    new_info = ask_why(story)
    print(new_info)
    print("\n\n-----------------------------------------------\n\n")

    new_info = new_info[-3:-1]
    new_story = add_new_info(story, new_info).replace("\n\n","\n")
    print(new_story)
    print("\n\n-----------------------------------------------\n\n")
