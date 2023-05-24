import pickle
import json
import random
import os, re
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from utils import *

with open("key.txt", 'r', encoding='utf-8') as f:
    keys = [i.strip() for i in f.readlines()]

def get_best_example(all_recommendations,n_best=1):
    final_scores = []
    for id in range(len(all_recommendations['plot_recommend'])):
        overall_score = 0
        for condition in all_recommendations:
            overall_score+= all_recommendations[condition][id][-1]
        final_scores.append((id,overall_score))
    best_ids = [i[0] for i in sorted(final_scores, key=lambda x: x[1], reverse=True)]
    if n_best == 1:
        return best_ids[0]
    else:
        return best_ids[0:n_best]
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

    #hits = sorted(hits, key=lambda x: x['score'], reverse=True)#[:k]
    scores = np.array([i['score'].item() for i in hits]).reshape(-1, 1)
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(scores)  # fit，在这里本质是生成min(x)和max(x)
    scalered_scores = scaler.transform(scores)  # 通过接口导出结果

    recommendation = [(i['corpus_id'],corpus[i['corpus_id']], scalered_scores[index][0]) for index,i in enumerate(hits)]
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
    hits = util.semantic_search(query_embeddings, corpus_embeddings,top_k=len(corpus))
    #print(hits)
    scores = np.array([i['score'] for i in hits[0]]).reshape(-1, 1)
    scaler = MinMaxScaler()  # 实例化
    scaler = scaler.fit(scores)  # fit，在这里本质是生成min(x)和max(x)
    scalered_scores = scaler.transform(scores)  # 通过接口导出结果

    recommendation = [(i['corpus_id'],corpus[i['corpus_id']], scalered_scores[index][0]) for index,i in enumerate(hits[0])]
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
    if 'an AI language model' in sentences[0]:
        sentences = sentences[1:]
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

    all_recommendations['mood_recommend'] = single_corpus_recommend('mood',queries['mood'], all_moods, k=topk)
    #all_recommendations['style_recommend'] = single_corpus_recommend('style',queries['style'], all_styles, k=topk)

    all_recommendations['genre_recommend'] = multi_corpus_recommend('genre',queries['genre'], all_genres, k=topk)
    all_recommendations['subject_recommend'] = multi_corpus_recommend('subjects',queries['subjects'], all_subjects, k=topk)
    all_recommendations['plot_recommend'] = multi_corpus_recommend('plots',queries['plots'], all_plots, k=topk)

    return all_recommendations

def ask_why(story,new_info=[None],plots=None,depth=1,width=2,unclarities=None):

    def summarize(story):
        prompt = 'Here is a story: "' + story +'"\n' + 'please point out '+str(width)+' major unclarities in the story in an organized list.\n'
        story_summary = generate(prompt)
        return story_summary

    if plots == None:
        unclarities = summarize(story)
        plots = clean_split(unclarities)
        new_info.append([(story,plots[0])])
        new_info.append([(story,plots[1])])

    new_plots = []
    for index,unclarity in enumerate(plots):
        prompt_1 = 'Here is a story: \n' + story +'\n' + 'An unclarity is: \n'+ unclarity +'\n' + 'Except for pure coincidence and mental reasons, please point out '+str(width)+' major reasons to rationalize the story. The additional information should be short and imply the topic: '+ list2text(queries['subjects']) + "."
        #prompt_1 = 'Here is a story: \n' + story +'\n' + 'An unclarity is: \n'+ unclarity +'\n' + 'Except for pure coincidence and subject reasons, reveal me some implicit background knowledge within one or two sentencese to rationalize the story. The additional information should be short and imply the topic: '+ list2text(queries['subjects']) + "."
        explanations = generate(prompt_1)
        explanations = clean_split(explanations)
        #prompt_2 = 'Here is a story: \n' + story +'\n' + 'Here is a plot: \n'+plot + '\n'+ 'Here are a list of possible reasons about why the above plot is reasonable in the story: \n' + list2text(explanations,type='phrase') + '\n' + ' pretend to be a professional writer, please select the reason that is the closest to the '+ list2text(queries['subjects']) + ' and only output the index number without any explanation.'
        #explanation_num = random.randint(0, len(explanations)-1)
        #output = generate(prompt_2)
        #explanation_num = int(re.sub("[/.,a-zA-Z]*","",output).strip()) - 1 # prompt 里explaination的序号 进行了+1
        new_plots+=explanations # pick an explanation
        for index,explanation in enumerate(explanations):
            current_index = len(new_info) + 1
            if current_index % width == 0:
                new_info.append(new_info[int(current_index / width)-1] + [(unclarity,explanation)])
            elif current_index % width !=0:
                new_info.append(new_info[int((current_index-1) / width)-1] + [(unclarity, explanation)])
            else:
                new_info.append([(unclarity, explanation)])

    # if depth < 1:
    #     depth += 1
    #     unclarities,new_info = ask_why(story, new_info, new_plots, depth,width, unclarities)
    #     return unclarities,new_info
    return unclarities,new_info

def add_new_info(story,picked_info):
    num_parts = 2
    new_story = []
    # reasons = [each[1] for each in new_info]
    # reasons = "\n".join(reasons)
    prompt = 'Here is a story: \n' + story +'\n' +'Here is some additional information: \n'+ picked_info + '\n'+ \
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
        question = 'Without further ado, start generating the second half of the story.' #分两步生成故事
        messages.append({"role": "assistant", "content": generation})
        messages.append({"role": "user", "content": question})

    new_story = "\n".join(new_story)
    return new_story
def pick_info(story,new_info):
    chain_of_reasons = []
    for index in range(3,7): #只选取最后四个info作为chain of reasons
        chain_of_reason = []
        for each in list(reversed(new_info[index])):
            chain_of_reason.append(each[-1])
        #chain_of_reason.append(each[0])
        chain_of_reason = list(reversed(chain_of_reason))
        chain_of_reasons.append(". ".join(chain_of_reason).replace("..","."))

    prompt = 'Here is a story: \n' + story + '\n' + 'Here are a list of reasons, explaining an unclarity in the story in depth: \n' + list2text(chain_of_reasons,type='phrase') + '\n' + 'Pretend to be a professional writer, please select the reason that is the closest to the '+ list2text(queries['subjects']) + ' and only output the index number without any explanation.'
    reason_id = int(re.sub("[/.,a-zA-Z]*","",generate(prompt)).strip()) - 1
    return chain_of_reasons[reason_id]

    #return picked_info
if __name__ == '__main__':
    queries = {'mood':'excited, thrilled','style':'magical','genre':['action'],'subjects':['adventure','death'],'plots':['he escaped from the island','he met a mermaid','he had a soup']}

    with open('../data/movie_data.json', 'r') as f:
        movie_data = json.load(f)

    all_recommendations = get_all_recommend(movie_data, queries,4)

    example_id = get_best_example(all_recommendations)

    # example_id_1 = all_recommendations['plot_recommend'][0][0]

    example = movie_data[example_id]

    examples = [example]


    prompt = make_prompt(examples=examples, conditions=queries)
    print(prompt)
    print("\n\n-----------------------------------------------\n\n")

    story = generate(prompt).replace("\n\n","\n")
    print(story)
    print("\n\n-----------------------------------------------\n\n")

    story_summary,new_info = ask_why(story)
    print(new_info)
    print("\n\n-----------------------------------------------\n\n")

    picked_info = pick_info(story,new_info)
    print(picked_info)
    print("\n\n-----------------------------------------------\n\n")

    new_story = add_new_info(story, picked_info).replace("\n\n","\n")
    print(new_story)
    print("\n\n-----------------------------------------------\n\n")
